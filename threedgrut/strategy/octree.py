from typing import Optional

import torch

from threedgrut.model.lod_model import MixtureOfGaussiansWithAnchor
from threedgrut.strategy.gs import GSStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import quaternion_to_so3, check_step_condition


class OctreeStrategy(GSStrategy):
    def __init__(self, config, model: MixtureOfGaussiansWithAnchor) -> None:
        super(GSStrategy, self).__init__(config, model)
        self.fork = self.conf.strategy.densify.fork
        self.update_ratio = self.conf.strategy.densify.update_ratio
        self.extra_ratio = self.conf.strategy.densify.extra_ratio
        self.extra_up = self.conf.strategy.densify.extra_up
        self.allow_overlap = self.conf.strategy.densify.allow_overlap
        self.densify_threshold = self.conf.strategy.densify.grad_threshold
        self.progressive = True
        self.coarse_intervals = []

        # Accumulation of the norms of the positions gradients
        self.densify_grad_norm_accum = torch.empty([0, 1])
        self.densify_grad_norm_denom = torch.empty([0, 1])
        self.anchor_mask = torch.empty([0, 1])

    def init_densification_buffer(self, checkpoint = None):
        super().init_densification_buffer(checkpoint)
        if checkpoint is not None:
            self.anchor_mask = checkpoint["anchor_mask"][0].detach()
        else:
            self.anchor_mask = torch.ones(self.model.num_gaussians, dtype=torch.bool)
            num_level = self.model.max_level - 1 - self.model.init_level
            if num_level > 0:
                q = 1 / self.conf.strategy.densify.coarse_factor
                a1 = self.conf.strategy.densify.coarse_iter*(1-q)/(1-q**num_level)
                temp_interval = 0
                for i in range(num_level):
                    interval = a1 * q ** i + temp_interval
                    temp_interval = interval
                    self.coarse_intervals.append(interval)


    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(self, sensor_position: torch.Tensor) -> None:
        params_grad = self.model.anchor.grad
        self.anchor_mask = (params_grad != 0).max(dim=1)[0]
        assert params_grad is not None
        distance_to_camera = (self.model.get_positions()[self.anchor_mask] - sensor_position).norm(dim=1, keepdim=True)

        self.densify_grad_norm_accum[self.anchor_mask] += (
            torch.norm(params_grad[self.anchor_mask] * distance_to_camera, dim=-1, keepdim=True) / 2
        )
        self.densify_grad_norm_denom[self.anchor_mask] += 1

    def post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""
        scene_updated = False
        # Densify the Gaussians

        if check_step_condition(step, self.conf.strategy.densify.start_iteration, self.conf.strategy.densify.end_iteration, self.conf.strategy.densify.frequency):
            self.densify_gaussians(step, scene_extent=scene_extent)
            scene_updated = True

        # # Prune the Gaussians based on their opacity
        # if check_step_condition(step, self.conf.strategy.prune.start_iteration, self.conf.strategy.prune.end_iteration, self.conf.strategy.prune.frequency):
        #     self.prune_gaussians_opacity()
        #     scene_updated = True

        # # Prune the Gaussians based on their scales
        # if check_step_condition(step, self.conf.strategy.prune_scale.start_iteration, self.conf.strategy.prune_scale.end_iteration, self.conf.strategy.prune_scale.frequency):
        #     self.prune_gaussians_scale(train_dataset)
        #     scene_updated = True

        # # Decay the density values
        # if check_step_condition(step, self.conf.strategy.density_decay.start_iteration, self.conf.strategy.density_decay.end_iteration, self.conf.strategy.density_decay.frequency):
        #     self.decay_density()

        # # Reset the Gaussian density
        # if check_step_condition(step, self.conf.strategy.reset_density.start_iteration, self.conf.strategy.reset_density.end_iteration, self.conf.strategy.reset_density.frequency):
        #     self.reset_density()

        return scene_updated

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, step, scene_extent):
        print("densification----------------------------------------")

        assert self.model.optimizer is not None, "Optimizer need to be initialized before splitting and cloning the Gaussians"
        densify_grad_norm = self.densify_grad_norm_accum / self.densify_grad_norm_denom
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0

        self.grow_anchor(
            iteration=step,
            anchor_grads=densify_grad_norm.squeeze(1),  # [N]
        )
        torch.cuda.empty_cache()
    
    @torch.cuda.nvtx.range("grow_anchor")
    def grow_anchor(self,
                    iteration: int,
                    anchor_grads: torch.Tensor):

        # 1) prune된 앵커들은 gradient 0으로
        anchor_grads = anchor_grads.clone()
        anchor_grads[~self.anchor_mask] = 0.0

        for cur_level in range(self.model.max_level):
            # 2) 현재 레벨 앵커
            level_mask = (self.model.get_levels().squeeze(1) == cur_level)
            if not level_mask.any():
                continue

            # 3) voxel 크기
            cur_size = self.model.voxel_size / (self.fork ** cur_level)
            ds_size = cur_size / self.fork

            # 4) 레벨별 threshold
            update_value    = self.fork ** self.update_ratio
            base            = self.densify_threshold * (update_value ** cur_level)
            next_threshold  = base * update_value
            extra_threshold = base * self.extra_ratio

            # 5) 같은 레벨 분할 후보
            candidate_same_level = (
                (anchor_grads >= base) &
                (anchor_grads < next_threshold) &
                level_mask
            )

            # 6) 하위 레벨 분할 후보
            candidate_down_level = torch.zeros_like(candidate_same_level)
            if cur_level < self.model.max_level - 1:
                candidate_down_level = (
                    (anchor_grads >= next_threshold) &
                    level_mask
                )

            # 7) extra_level 업데이트
            if (not self.progressive) or (iteration > self.coarse_intervals[-1]):
                extra_mask = (anchor_grads >= extra_threshold) & level_mask
                self.model.extra_level[extra_mask] += extra_up

            # ------------------------------------------------------------
            # 8) 같은 레벨 후보 앵커 좌표 뽑기
            cand_same_coords = self.model.get_anchor()[candidate_same_level]  # [M1,3]

            # 9) 하위 레벨 후보 앵커 좌표 뽑기
            cand_down_coords = self.model.get_anchor()[candidate_down_level]  # [M2,3]

            # ------------------------------------------------------------
            # 10) grid cell 계산 및 중복 처리 + weed_out (same 레벨)
            grid_coords = torch.round(
                (self.model.get_anchor()[level_mask] - self.model.init_pos) / cur_size - self.model.padding
            ).int()  # [num_level,3]

            selected_grid_coords = torch.round(
                (cand_same_coords - self.model.init_pos) / cur_size - self.model.padding
            ).int()  # [M1,3]
            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )  # [U1,3], [M1]

            if self.allow_overlap:
                remove_dup = torch.ones(
                    (selected_grid_coords_unique.shape[0],),
                    dtype=torch.bool,
                    device="cuda"
                )
                candidate_anchor = (
                    selected_grid_coords_unique.float() * cur_size
                    + self.model.init_pos
                    + self.model.padding * cur_size
                )  # [U1,3]
                new_level = torch.ones(
                    (candidate_anchor.shape[0],),
                    dtype=torch.int,
                    device="cuda"
                ) * cur_level  # [U1]

                candidate_anchor, new_level, _, weed_mask = self.model.weed_out(
                    candidate_anchor, new_level
                )
                remove_dup_clone = remove_dup.clone()
                remove_dup[remove_dup_clone] = weed_mask

            elif (selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0):
                remove_dup_init = self.model.get_remove_duplicates(
                    grid_coords, selected_grid_coords_unique
                )  # [U1]
                remove_dup = ~remove_dup_init  # [U1]
                candidate_anchor = (
                    selected_grid_coords_unique[remove_dup].float() * cur_size
                    + self.model.init_pos
                    + self.model.padding * cur_size
                )  # [U1',3]
                new_level = torch.ones(
                    (candidate_anchor.shape[0],),
                    dtype=torch.int,
                    device="cuda"
                ) * cur_level  # [U1']

                candidate_anchor, new_level, _, weed_mask = self.weed_out(
                    candidate_anchor, new_level
                )
                remove_dup_clone = remove_dup.clone()
                remove_dup[remove_dup_clone] = weed_mask

            else:
                candidate_anchor = torch.zeros(
                    (0, 3), dtype=torch.float, device="cuda"
                )
                remove_dup = torch.zeros(
                    (selected_grid_coords_unique.shape[0],),
                    dtype=torch.bool,
                    device="cuda"
                )
                new_level = torch.zeros(
                    (0,), dtype=torch.int, device="cuda"
                )

            # ------------------------------------------------------------
            # 11) 하위 레벨 분할 후보에 대해서도 같은 처리
            level_ds_mask = (self.model.get_levels().squeeze(1) == (cur_level + 1))
            grid_coords_ds = torch.round(
                (self.model.get_anchor()[level_ds_mask] - self.model.init_pos) / ds_size - self.model.padding
            ).int()  # [num_level+1,3]

            selected_grid_coords_ds = torch.round(
                (cand_down_coords - self.model.init_pos) / ds_size - self.model.padding
            ).int()  # [M2,3]
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(
                selected_grid_coords_ds, return_inverse=True, dim=0
            )  # [U2,3], [M2]

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and (cur_level < self.model.max_level - 1):
                if self.allow_overlap:
                    remove_dup_ds = torch.ones(
                        (selected_grid_coords_unique_ds.shape[0],),
                        dtype=torch.bool, device="cuda"
                    )
                    candidate_anchor_ds = (
                        selected_grid_coords_unique_ds.float() * ds_size
                        + self.model.init_pos
                        + self.model.padding * ds_size
                    )  # [U2,3]
                    new_level_ds = torch.ones(
                        (candidate_anchor_ds.shape[0],),
                        dtype=torch.int, device="cuda"
                    ) * (cur_level + 1)  # [U2]

                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.model.weed_out(
                        candidate_anchor_ds, new_level_ds
                    )
                    remove_dup_ds_clone = remove_dup_ds.clone()
                    remove_dup_ds[remove_dup_ds_clone] = weed_ds_mask

                elif (selected_grid_coords_unique_ds.shape[0] > 0 and
                    grid_coords_ds.shape[0] > 0):
                    remove_dup_ds_init = self.model.get_remove_duplicates(
                        grid_coords_ds, selected_grid_coords_unique_ds
                    )  # [U2]
                    remove_dup_ds = ~remove_dup_ds_init  # [U2]
                    candidate_anchor_ds = (
                        selected_grid_coords_unique_ds[remove_dup_ds].float() * ds_size
                        + self.model.init_pos
                        + self.model.padding * ds_size
                    )  # [U2',3]
                    new_level_ds = torch.ones(
                        (candidate_anchor_ds.shape[0],),
                        dtype=torch.int, device="cuda"
                    ) * (cur_level + 1)  # [U2']

                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.model.weed_out(
                        candidate_anchor_ds, new_level_ds
                    )
                    remove_dup_ds_clone = remove_dup_ds.clone()
                    remove_dup_ds[remove_dup_ds_clone] = weed_ds_mask

                else:
                    candidate_anchor_ds = torch.zeros(
                        (0, 3), dtype=torch.float, device="cuda"
                    )
                    remove_dup_ds = torch.zeros(
                        (selected_grid_coords_unique_ds.shape[0],),
                        dtype=torch.bool, device="cuda"
                    )
                    new_level_ds = torch.zeros(
                        (0,), dtype=torch.int, device="cuda"
                    )
            else:
                candidate_anchor_ds = torch.zeros(
                    (0, 3), dtype=torch.float, device="cuda"
                )
                remove_dup_ds = torch.zeros(
                    (selected_grid_coords_unique_ds.shape[0],),
                    dtype=torch.bool, device="cuda"
                )
                new_level_ds = torch.zeros(
                    (0,), dtype=torch.int, device="cuda"
                )

            # ------------------------------------------------------------
            # 12) 새로운 앵커가 없으면 넘어감
            if (candidate_anchor.shape[0] + candidate_anchor_ds.shape[0]) == 0:
                continue

            # 13) 새 앵커 좌표, 레벨 합치기
            new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)  # [M_new,3]
            new_level  = torch.cat([new_level, new_level_ds], dim=0).unsqueeze(1).float().cuda()  # [M_new,1]
            M_new = new_anchor.shape[0]

            # --------------------------------------------------------
            # 14) 새 SH feature (k=1)이므로 간단히 Boolean 인덱싱
            combined_mask = (candidate_same_level | candidate_down_level)  # [N]
            new_features = self.model.get_features()[combined_mask]               # [M_new, F, 3]
            new_features_albedo   = new_features[:1, :]   # [M_new,1,3]
            new_features_specular = new_features[1: :]    # [M_new,F-1,3]

            # --------------------------------------------------------
            # 15) 새 Opacity 초기화
            new_density = self.model.density_activation_inv(
                0.1 * torch.ones((M_new, 1), dtype=torch.float, device="cuda")
            )  # [M_new,1]

            # --------------------------------------------------------
            # 16) 새 Scale 초기화 (3차원)
            new_scale = self.model.scale_activation_inv(
                cur_size * torch.ones((M_new, 3), device="cuda")
            )  # [M_new,3]

            # --------------------------------------------------------
            # 17) 새 Rotation 초기화 (identity quaternion)
            new_rotation = torch.zeros((M_new, 4), device="cuda")  # [M_new,4]
            new_rotation[:, 0] = 1.0

            # --------------------------------------------------------
            # 18) 새 Offset 초기화 (k=1이므로 [0,0,0])
            new_offset = torch.zeros((M_new, 3), device="cuda")  # [M_new,3]
            new_offset_scale = self.model.scale_activation_inv(torch.ones((M_new, 3), device="cuda"))  # [M_new,3]

            # --------------------------------------------------------
            # 19) 새 extra_level 초기화
            new_extra_level = torch.zeros((M_new,), device="cuda")  # [M_new]

            # --------------------------------------------------------
            # 20) anchor_demon, opacity_accum, anchor_mask 확장
            self.anchor_demon = torch.cat([
                self.anchor_demon,
                torch.zeros((M_new, 1), device="cuda")
            ], dim=0)  # [N+M_new,1]

            self.opacity_accum = torch.cat([
                self.opacity_accum,
                torch.zeros((M_new, 1), device="cuda")
            ], dim=0)  # [N+M_new,1]

            new_mask = torch.ones((M_new,), dtype=torch.bool, device="cuda")
            self.anchor_mask = torch.cat([self.anchor_mask, new_mask], dim=0)  # [N+M_new]

            torch.cuda.empty_cache()

            # --------------------------------------------------------
            # 21) 파라미터 텐서 전체 업데이트 (cat_tensors_to_optimizer 사용)
            d = {
                "anchor"       : new_anchor,         # [M_new,3]
                "scale"        : new_scale,        # [M_new,3]
                "rotation"     : new_rotation,       # [M_new,4]
                "features_albedo"  : new_features_albedo,    # [M_new,1,3]
                "features_specular": new_features_specular,  # [M_new,F-1,3]
                "offset"       : new_offset,        # [M_new,3]
                "offset_scale"       : new_offset_scale,        # [M_new,3]
                "density"      : new_density       # [M_new,1]
            }

            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self.model.anchor        = optimizable_tensors["anchor"]        # [N+M_new,3]
            self.model.scale       = optimizable_tensors["scaling"]       # [N+M_new,3]
            self.model.rotation      = optimizable_tensors["rotation"]      # [N+M_new,4]
            self.model.features_albedo   = optimizable_tensors["features_dc"]   # [N+M_new,1,3]
            self.model.features_specular = optimizable_tensors["features_rest"] # [N+M_new,F-1,3]
            self.model.offset        = optimizable_tensors["offset"]        # [N+M_new,3]
            self.model.offset_scale        = optimizable_tensors["scale"]        # [N+M_new,3]
            self.model.density       = optimizable_tensors["density"]       # [N+M_new,1]

            self.model.level       = torch.cat([self.model.level, new_level], dim=0)       # [N+M_new,1]
            self.model.extra_level = torch.cat([self.model.extra_level, new_extra_level], dim=0)  # [N+M_new]
