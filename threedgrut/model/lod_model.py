import os, math
import torch
from torch_scatter import scatter_max
import numpy as np
from plyfile import PlyData, PlyElement

import threedgrt_tracer
from threedgrut.datasets.utils import read_next_bytes, read_colmap_points3D_text
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    get_activation_function,
    get_scheduler,
    sh_degree_to_num_features,
    sh_degree_to_specular_dim,
    to_np,
    to_torch,
    quaternion_to_so3,
)
from threedgrut.utils.render import RGB2SH, SH2RGB
from threedgrut.utils.knn import knn


class MixtureOfGaussiansWithAnchor(MixtureOfGaussians):
    @property
    def num_gaussians(self):
        return self.density.shape[0]

    def get_positions(self) -> torch.Tensor:
        return self.get_anchor() + self.get_offset() * self.get_offset_scale()

    def get_anchor(self) -> torch.Tensor:
        return self.anchor

    def get_offset(self) -> torch.Tensor:
        return self.offset

    def get_offset_scale(self, preactivation=False) -> torch.Tensor:
        if preactivation:
            return self.offset_scale
        else:
            return self.scale_activation(self.offset_scale)

    def get_levels(self) -> torch.Tensor:
        return self.level

    def get_extra_levels(self) -> torch.Tensor:
        return self.extra_level

    def get_num_gaussians(self) -> int:
        return len(self.density)

    def get_anchor_masks(self) -> torch.Tensor:
        return self.anchor_mask

    def map_to_int_level(self, pred_level, cur_level):
        int_level = torch.round(pred_level).int()
        int_level = torch.clamp(int_level, min=0, max=cur_level)
        return int_level

    def get_model_parameters(self) -> dict:
        params = super().get_model_parameters()
        del params["positions"]

        params["anchor"] = self.anchor
        params["offset"] = self.offset
        params["offset_scale"] = self.offset_scale
        params["level"] = self.level
        params["extra_level"] = self.extra_level
        params["std_dist"] = self.std_dist
        return params

    def __init__(self, conf, scene_extent=None):
        # Rendering method
        if conf.render.method == "3dgut":
            raise ValueError(f"Unsupported type with lod!")

        super().__init__(conf, scene_extent)
        self.renderer = threedgrt_tracer.LoDTracer(conf)
        if "positions" in self._parameters:
            del self._parameters["positions"]

        self.anchor = torch.nn.Parameter(torch.empty([0, 3]))
        self.offset = torch.nn.Parameter(torch.empty([0, 3]))
        self.offset_scale = torch.nn.Parameter(torch.empty([0, 3]))
        self.levels = torch.nn.Parameter(torch.empty([0, 1]), requires_grad=False)
        self.extra_levels = torch.nn.Parameter(torch.empty([0, 1]), requires_grad=False)
        self.std_dist = 0

        # hyper parameter
        self.max_level = self.conf.model.get("max_level", -1)
        self.init_level = self.conf.model.get("init_level", -1)
        self.extend = self.conf.model.get("extend", -1)
        self.base_layer = self.conf.model.get("base_layer", -1)
        self.fork = self.conf.model.get("fork", 2)
        self.default_voxel_size = self.conf.model.get("default_voxel_size", -1)
        self.padding = self.conf.model.get("padding", None)
        self.visible_threshold = self.conf.model.get("visible_threshold", None)
        self.dist_ratio = self.conf.model.get("dist_ratio", None)
        # self.n_offsets = self.conf.model.n_offsets

    def setup_scheduler(self):
        self.schedulers = {}
        for name, args in self.conf.scheduler.items():
            if args.type is not None and getattr(self, name).requires_grad:
                if name == "anchor" or name == "offset":
                    self.schedulers[name] = get_scheduler(args.type)(
                        lr_init=args.lr_init * self.scene_extent,
                        lr_final=args.lr_final * self.scene_extent,
                        max_steps=args.max_steps,
                    )
                else:
                    self.schedulers[name] = get_scheduler(args.type)(**args)

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer=True):
        self.anchor = checkpoint["anchor"]
        self.offset = checkpoint["offset"]
        self.offset_scale = checkpoint["offset_scale"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.features_albedo = checkpoint["features_albedo"]
        self.features_specular = checkpoint["features_specular"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.scene_extent = checkpoint["scene_extent"]
        self.level = checkpoint["level"]
        self.extra_level = checkpoint["extra_level"]
        self.std_dist = checkpoint["std_dist"]
        logger.info(f"# of gaussians: {self.num_gaussians}")

        if self.progressive_training:
            self.feature_dim_increase_interval = checkpoint[
                "feature_dim_increase_interval"
            ]
            self.feature_dim_increase_step = checkpoint["feature_dim_increase_step"]

        self.background.load_state_dict(checkpoint["background"])
        if setup_optimizer:
            self.set_optimizable_parameters()
            self.setup_optimizer(state_dict=checkpoint["optimizer"])
        self.validate_fields()

    def validate_fields(self):
        num_gaussians = self.num_gaussians
        # assert self.positions.shape == (num_gaussians, 3)
        # assert self.anchor.shape == (num_gaussians, 3)
        # assert self.offset.shape == (num_gaussians, 3)
        assert self.density.shape == (num_gaussians, 1)
        assert self.rotation.shape == (num_gaussians, 4)
        assert self.scale.shape == (num_gaussians, 3)

        if self.feature_type == "sh":
            assert self.features_albedo.shape == (num_gaussians, 3)
            specular_sh_dims = sh_degree_to_specular_dim(self.max_n_features)
            assert self.features_specular.shape == (num_gaussians, specular_sh_dims)
        else:
            raise ValueError("Neural features not yet supported.")

    def set_level(self, points, cameras):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()

        for cam in cameras:
            cam_info = torch.tensor((cam[0], cam[1], cam[2], 1)).float().cuda()
            self.cam_infos = torch.cat(
                (self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0
            )
            dist = torch.sqrt(torch.sum((points - cam) ** 2, dim=1))
            dist_max = torch.quantile(dist, self.dist_ratio)
            dist_min = torch.quantile(dist, 1 - self.dist_ratio)
            new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
            new_dist = new_dist
            all_dist = torch.cat((all_dist, new_dist), dim=0)

        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.std_dist = float(dist_max)
        if self.max_level == -1:
            self.max_level = (
                torch.round(torch.log2(dist_max / dist_min) / math.log2(self.fork))
                .int()
                .item()
                + 1
            )
        if self.init_level == -1:
            self.init_level = int(self.max_level / 2)

    def octree_sample(self, points, colors):
        torch.cuda.synchronize()
        self.pos = torch.empty(0, 3).float().cuda()
        self.colors = torch.empty(0, 3).float().cuda()
        self.level = torch.empty(0).int().cuda()

        for cur_level in range(self.max_level):
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            new_candidates = torch.round((points - self.init_pos) / cur_size)
            new_candidates_unique, inverse_indices = torch.unique(
                new_candidates, return_inverse=True, dim=0
            )
            new_positions = new_candidates_unique * cur_size + self.init_pos
            new_positions += self.padding * cur_size
            new_levels = (
                torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda")
                * cur_level
            )
            new_colors = scatter_max(
                colors, inverse_indices.unsqueeze(1).expand(-1, colors.size(1)), dim=0
            )[0]
            self.pos = torch.concat((self.pos, new_positions), dim=0)
            self.colors = torch.concat((self.colors, new_colors), dim=0)
            self.level = torch.concat((self.level, new_levels), dim=0)
            logger.info(
                f"[Level: {cur_level}] size: {cur_size}, count: {new_positions.shape[0]}"
            )
        torch.cuda.synchronize()

    def weed_out(self, gaussian_positions, gaussian_levels):
        visible_count = torch.zeros(
            gaussian_positions.shape[0], dtype=torch.int, device="cuda"
        )
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = (
                torch.sqrt(torch.sum((gaussian_positions - cam_center) ** 2, dim=1))
                * scale
            )
            pred_level = torch.log2(self.std_dist / dist) / math.log2(self.fork)
            int_level = self.map_to_int_level(pred_level, self.max_level - 1)
            visible_count += (gaussian_levels <= int_level).int()
        visible_count = visible_count / len(self.cam_infos)
        weed_mask = visible_count > self.visible_threshold
        mean_visible = torch.mean(visible_count)
        return (
            gaussian_positions[weed_mask],
            gaussian_levels[weed_mask],
            mean_visible,
            weed_mask,
        )

    def default_initialize_from_points(
        self, pts, observer_pts, colors=None, use_observer_pts=True
    ):
        logger.info(f"Generating Octree...")
        self.set_level(pts, observer_pts)
        # unknown
        # self.spatial_lr_scale = spatial_lr_scale

        box_min = torch.min(pts) * self.extend
        box_max = torch.max(pts) * self.extend
        box_d = box_max - box_min

        if self.base_layer < 0:
            self.base_layer = (
                torch.round(torch.log2(box_d / self.default_voxel_size)).int().item()
                - (self.max_level // 2)
                + 1
            )

        self.voxel_size = box_d / (float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(pts, colors)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.pos, self.level, self.visible_threshold, _ = self.weed_out(
                self.pos, self.level
            )
        self.pos, self.level, _, weed_mask = self.weed_out(self.pos, self.level)
        self.colors = self.colors[weed_mask]

        logger.info(f"Branches of Tree: {self.fork}")
        logger.info(f"Base Layer of Tree: {self.base_layer}")
        logger.info(f"Visible Threshold: {self.visible_threshold}")
        logger.info(f"LOD Levels: {self.max_level}")
        logger.info(f"Initial Levels: {self.init_level}")
        logger.info(f"Initial Voxel Number: {self.pos.shape[0]}")
        logger.info(f"Min Voxel Size: {self.voxel_size/(2.0 ** (self.max_level - 1))}")
        logger.info(f"Max Voxel Size: {self.voxel_size}")

        fused_point_cloud, fused_color = self.pos, RGB2SH(self.colors / 255.0)
        n_features = sh_degree_to_specular_dim(self.max_n_features)
        offsets = torch.zeros((fused_point_cloud.shape[0], 3)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3 + n_features)).float().cuda()
        features[:, :3] = fused_color
        features_albedo, features_specular = torch.split(
            features, [3, n_features], dim=1
        )

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1
        opacities = self.density_activation_inv(
            self.conf.model.default_density
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device
            )
        )

        self.anchor = torch.nn.Parameter(fused_point_cloud)
        self.offset = torch.nn.Parameter(offsets)
        self.offset_scale = torch.nn.Parameter(scales[:, :3])
        self.features_albedo = torch.nn.Parameter(features_albedo)
        self.features_specular = torch.nn.Parameter(features_specular)
        self.scale = torch.nn.Parameter(scales[:, 3:])
        self.rotation = torch.nn.Parameter(rots)
        self.density = torch.nn.Parameter(opacities)
        self.level = self.level.unsqueeze(dim=1).float()
        self.extra_level = torch.zeros(
            self.num_gaussians, dtype=torch.float, device=self.device
        )
        self.anchor_mask = torch.ones(
            self.num_gaussians, dtype=torch.bool, device=self.device
        )

        self.positions = self.anchor + self.offset * torch.exp(self.offset_scale)

        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()

    @torch.no_grad()
    def init_from_ply(self, mogt_path: str, init_model=True):
        plydata = PlyData.read(mogt_path)
        v = plydata.elements[0]

        for line in plydata.obj_info:
            key, value = line.split()
            if key == "standard_dist":
                self.std_dist = float(value)
                break

        # 1. anchor, offset
        mogt_anchor = np.stack(
            (np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])), axis=1
        )
        num_gaussians = mogt_anchor.shape[0]

        mogt_offset = np.zeros((num_gaussians, 3))
        mogt_offset[:, 0] = np.asarray(v["f_offset_0"])
        mogt_offset[:, 1] = np.asarray(v["f_offset_1"])
        mogt_offset[:, 2] = np.asarray(v["f_offset_2"])

        # 2. density
        mogt_densities = np.asarray(v["opacity"])[..., np.newaxis]

        # 3. albedo
        mogt_albedo = np.zeros((num_gaussians, 3))
        mogt_albedo[:, 0] = np.asarray(v["f_dc_0"])
        mogt_albedo[:, 1] = np.asarray(v["f_dc_1"])
        mogt_albedo[:, 2] = np.asarray(v["f_dc_2"])

        # 4. specular
        extra = sorted(
            [p.name for p in v.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        num_spec = (self.max_n_features + 1) ** 2 - 1
        mogt_spec = np.zeros((num_gaussians, len(extra)))
        for i, name in enumerate(extra):
            mogt_spec[:, i] = np.asarray(v[name])
        mogt_spec = (
            mogt_spec.reshape((num_gaussians, 3, num_spec))
            .transpose(0, 2, 1)
            .reshape((num_gaussians, num_spec * 3))
        )

        # 5. scale: exp(scale_0~5)
        scale_names = [p.name for p in v.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        mogt_scales = np.zeros((num_gaussians, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            mogt_scales[:, idx] = np.asarray(v[attr_name])

        # 6. rotation
        rot_names = [p.name for p in v.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        mogt_rotation = np.zeros((num_gaussians, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            mogt_rotation[:, idx] = np.asarray(v[attr_name])

        # 7. level
        mogt_level = np.asarray(v["level"])[..., np.newaxis]
        mogt_extra_level = np.asarray(v["extra_level"])[..., np.newaxis]

        t = torch.tensor
        dev = self.device
        self.anchor = torch.nn.Parameter(
            t(mogt_anchor, dtype=self.anchor.dtype, device=dev)
        )
        self.offset = torch.nn.Parameter(
            t(mogt_offset, dtype=self.anchor.dtype, device=dev)
        )
        self.offset_scale = torch.nn.Parameter(
            t(mogt_scales[:, :3], dtype=self.anchor.dtype, device=dev)
        )
        self.density = torch.nn.Parameter(
            t(mogt_densities, dtype=self.density.dtype, device=dev)
        )
        self.features_albedo = torch.nn.Parameter(
            t(mogt_albedo, dtype=self.features_albedo.dtype, device=dev)
        )
        self.features_specular = torch.nn.Parameter(
            t(mogt_spec, dtype=self.features_specular.dtype, device=dev)
        )
        self.scale = torch.nn.Parameter(
            t(mogt_scales[:, 3:], dtype=self.scale.dtype, device=dev)
        )
        self.rotation = torch.nn.Parameter(
            t(mogt_rotation, dtype=self.rotation.dtype, device=dev)
        )
        self.level = torch.nn.Parameter(
            t(mogt_level, dtype=self.levels.dtype, device=dev), requires_grad=False
        )
        self.extra_level = torch.nn.Parameter(
            t(mogt_extra_level, dtype=self.extra_levels.dtype, device=dev),
            requires_grad=False,
        )
        self.anchor_mask = torch.ones(
            self.num_gaussians, dtype=torch.bool, device="cuda"
        )
        self.positions = t(mogt_anchor + mogt_offset * np.exp(mogt_scales[:, :3]))
        self.n_active_features = self.max_n_features

        if init_model:
            self.set_optimizable_parameters()
            self.setup_optimizer()
            self.validate_fields()

    @torch.no_grad()
    def init_from_random_point_cloud(
        self,
        observer_pts,
        xyz_max: torch.Tensor,
        xyz_min: torch.Tensor,
        num_gaussians: int = 100_000,
        dtype=torch.float32,
        set_optimizable_parameters: bool = True,
    ):
        logger.info(f"Generating random point cloud ({num_gaussians})...")
        logger.info(f"{xyz_min.shape} {xyz_max.shape}")

        fused_point_cloud = (
            torch.rand((num_gaussians, 3), dtype=dtype, device=self.device)
            * (xyz_max - xyz_min)
            + xyz_min
        )

        # sh albedo in [0, 0.0039]
        fused_color = (
            SH2RGB(
                (
                    torch.rand((num_gaussians, 3), dtype=dtype, device=self.device)
                    / 255.0
                )
            )
            * 255.0
        )

        return self.default_initialize_from_points(
            fused_point_cloud, observer_pts, fused_color
        )

    def init_from_initial_point_cloud(self, path, observer_pts):
        plydata = PlyData.read(path)
        v = plydata.elements[0]

        fused_point_cloud = torch.tensor(
            np.stack((v["x"], v["y"], v["z"]), axis=1), device=self.device
        )

        # fused_color = torch.tensor(
        #     np.stack((v["red"], v["green"], v["blue"]), axis=1),
        #     dtype=torch.uint8,
        #     device=self.device,
        # )
        fused_color = torch.ones_like(fused_point_cloud, dtype=torch.uint8) * 255

        return self.default_initialize_from_points(
            fused_point_cloud, observer_pts, fused_color
        )
