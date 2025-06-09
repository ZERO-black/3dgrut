from typing import Optional

import torch

from threedgrut.model.lod_model import MixtureOfGaussiansWithAnchor
from threedgrut.strategy.gs import GSStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import quaternion_to_so3, check_step_condition
import math

class OctreeStrategy(GSStrategy):
    def __init__(self, config, model: MixtureOfGaussiansWithAnchor) -> None:
        super(GSStrategy, self).__init__(config, model)
        self.fork = self.conf.strategy.densify.fork
        self.update_ratio = self.conf.strategy.densify.update_ratio
        self.extra_ratio = self.conf.strategy.densify.extra_ratio
        self.extra_up = self.conf.strategy.densify.extra_up
        self.allow_overlap = self.conf.strategy.densify.allow_overlap
        self.densify_threshold = self.conf.strategy.densify.grad_threshold
        self.prune_density_threshold = self.conf.strategy.prune.density_threshold
        self.new_max_density = self.conf.strategy.reset_density.new_max_density

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
        # if check_step_condition(
        #     step,
        #     self.conf.strategy.prune.start_iteration,
        #     self.conf.strategy.prune.end_iteration,
        #     self.conf.strategy.prune.frequency,
        # ):
        #     self.prune_gaussians_opacity()
        #     scene_updated = True

        # # # Prune the Gaussians based on their scales
        # if check_step_condition(
        #     step,
        #     self.conf.strategy.prune_scale.start_iteration,
        #     self.conf.strategy.prune_scale.end_iteration,
        #     self.conf.strategy.prune_scale.frequency,
        # ):
        #     self.prune_gaussians_scale(train_dataset)
        #     scene_updated = True

        # # Decay the density values
        # if check_step_condition(
        #     step,
        #     self.conf.strategy.density_decay.start_iteration,
        #     self.conf.strategy.density_decay.end_iteration,
        #     self.conf.strategy.density_decay.frequency,
        # ):
        #     self.decay_density()

        # # Reset the Gaussian density
        # if check_step_condition(
        #     step,
        #     self.conf.strategy.reset_density.start_iteration,
        #     self.conf.strategy.reset_density.end_iteration,
        #     self.conf.strategy.reset_density.frequency,
        # ):
        #     self.reset_density()

        return scene_updated

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, step, scene_extent):
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

        total = self.anchor_mask.numel()
        valid = self.anchor_mask.sum().item()
        logger.info(f"valid offsets: {valid}/{total} ({valid/total*100:.2f}%)")
        g = anchor_grads
        logger.info(f"mean: {g.mean().item()}")
        logger.info(f"std: {g.std().item()}")
        logger.info(f"min: {g.min().item()}, max: {g.max().item()}")
        logger.info(f"median: {g.median().item()}")

        # 1) prune된 앵커들은 gradient 0으로
        anchor_grads = anchor_grads.clone()
        anchor_grads[~self.anchor_mask] = 0.0
        init_shape = self.model.num_gaussians

        for cur_level in range(self.model.max_level):
            # 2) 현재 레벨 앵커
            level_mask = (self.model.get_levels().squeeze(1) == cur_level)
            if not level_mask.any():
                if self.conf.strategy.print_stats:
                    logger.info(f"[Level: {cur_level}] 0 gaussians")
                continue

            # 3) voxel 크기
            cur_size = self.model.voxel_size / (self.fork ** cur_level)
            ds_size = cur_size / self.fork

            # 4) 레벨별 threshold
            update_value = self.fork**self.update_ratio
            base            = self.densify_threshold * (update_value ** cur_level)
            next_threshold  = base * update_value
            extra_threshold = base * self.extra_ratio

            scales = self.model.get_scale()  # shape: (N, 3)
            large_scale_mask = (scales > 2 * cur_size).any(dim=1) & level_mask

            # with torch.no_grad():
            #     # ln(2)를 빼서 exp(raw_scale - ln(2)) = exp(raw_scale)/2
            #     self.model.scale.data[large_scale_mask] -= math.log(2)

            # 5) 같은 레벨 분할 후보
            candidate_same_level = (
                (anchor_grads >= base) & (anchor_grads < next_threshold) & level_mask
            ) | large_scale_mask

            # 6) 하위 레벨 분할 후보
            candidate_down_level = torch.zeros_like(candidate_same_level)
            if cur_level < self.model.max_level - 1:
                candidate_down_level = (anchor_grads >= next_threshold) & level_mask
            logger.info(
                f"[Level: {cur_level}] curr: {torch.sum(level_mask)} same: {torch.sum(candidate_same_level)} down: {torch.sum(candidate_down_level)}"
            )

            # 7) extra_level 업데이트
            if (not self.progressive) or (iteration > self.coarse_intervals[-1]):
                extra_mask = (anchor_grads >= extra_threshold) & level_mask
                self.model.extra_level[extra_mask] += self.extra_up

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
            unique_mask = torch.zeros(selected_grid_coords.shape[0], dtype=torch.bool)
            unique_mask[inverse_indices] = True
            selected_features_unique = self.model.get_features()[candidate_same_level][
                unique_mask
            ]

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
                selected_features_unique = selected_features_unique[weed_mask]

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

                candidate_anchor, new_level, _, weed_mask = self.model.weed_out(
                    candidate_anchor, new_level
                )
                remove_dup_clone = remove_dup.clone()
                remove_dup[remove_dup_clone] = weed_mask
                selected_features_unique = selected_features_unique[weed_mask]
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
                selected_features_unique = torch.zeros(
                    (0, 3), dtype=torch.float, device="cuda"
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
            logger.info(
                f"cur_level={cur_level}  ds raw candidates: {cand_down_coords.shape[0]}"
            )

            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(
                selected_grid_coords_ds, return_inverse=True, dim=0
            )  # [U2,3], [M2]
            logger.info(
                f"cur_level={cur_level}  unique ds coords: {selected_grid_coords_unique_ds.shape[0]}"
            )

            unique_mask = torch.zeros(
                selected_grid_coords_ds.shape[0], dtype=torch.bool
            )
            unique_mask[inverse_indices_ds] = True

            selected_features_unique_ds = self.model.get_features()[
                candidate_down_level
            ][unique_mask]

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

                    candidate_anchor_ds, new_level_ds, mean_vis, weed_ds_mask = (
                        self.model.weed_out(candidate_anchor_ds, new_level_ds)
                    )
                    # logger.info(
                    #     f"cur_level={cur_level} ds before weed_out: {candidate_anchor_ds.shape[0]} → after: {weed_ds_mask.sum().item()}, mean_visible={mean_vis.item():.3f}"
                    # )
                    remove_dup_ds_clone = remove_dup_ds.clone()
                    remove_dup_ds[remove_dup_ds_clone] = weed_ds_mask
                    selected_features_unique_ds = selected_features_unique_ds[
                        weed_ds_mask
                    ]
                    # logger.info(
                    #     f"cur_level={cur_level}  ds survived after model.weed_out: {weed_ds_mask.sum().item()}"
                    # )
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

                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = (
                        self.model.weed_out(candidate_anchor_ds, new_level_ds)
                    )

                    remove_dup_ds_clone = remove_dup_ds.clone()
                    remove_dup_ds[remove_dup_ds_clone] = weed_ds_mask
                    selected_features_unique_ds = selected_features_unique_ds[
                        weed_ds_mask
                    ]
                    # logger.info(
                    #     f"cur_level={cur_level}  ds survived after model.weed_out: {weed_ds_mask.sum().item()}"
                    # )
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
                    selected_features_unique_ds = torch.zeros(
                        (0, 3), dtype=torch.float, device="cuda"
                    )
                    logger.info(f"cur_level={cur_level}  ds branch skipped")
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
                selected_features_unique_ds = torch.zeros(
                    (0, 3), dtype=torch.float, device="cuda"
                )

            # ------------------------------------------------------------
            # 12) 새로운 앵커가 없으면 넘어감
            if (candidate_anchor.shape[0] + candidate_anchor_ds.shape[0]) == 0:
                continue

            # 13) 새 앵커 좌표, 레벨 합치기
            new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)  # [M_new,3]
            new_level  = torch.cat([new_level, new_level_ds], dim=0).unsqueeze(1).float().cuda()  # [M_new,1]
            M_new = new_anchor.shape[0]
            logger.info(
                f"cur_level={cur_level}, same={candidate_anchor.shape[0]} down={candidate_anchor_ds.shape[0]}"
            )

            # --------------------------------------------------------
            # 14) 새 SH feature (k=1)이므로 간단히 Boolean 인덱싱
            new_features = torch.cat(
                [selected_features_unique, selected_features_unique_ds], dim=0
            )  # [M_new, F, 3]
            new_features_albedo = new_features[:, :3]  # [M_new,1,3]
            new_features_specular = new_features[:, 3:]  # [M_new,F-1,3]

            # --------------------------------------------------------
            # 15) 새 Opacity 초기화
            new_density = self.model.density_activation_inv(
                0.1 * torch.ones((M_new, 1), dtype=torch.float, device="cuda")
            )  # [M_new,1]

            # --------------------------------------------------------
            # 16) 새 Scale 초기화 (3차원)
            # new_scale = self.model.scale_activation_inv(
            #     cur_size * torch.ones((M_new, 3), device="cuda")
            # )  # [M_new,3]
            new_scale = self.model.scale_activation_inv(
                torch.cat(
                    [
                        cur_size * torch.ones_like(candidate_anchor, device="cuda"),
                        ds_size * torch.ones_like(candidate_anchor_ds, device="cuda"),
                    ],
                    dim=0,
                )
            )

            # --------------------------------------------------------
            # 17) 새 Rotation 초기화 (identity quaternion)
            new_rotation = torch.zeros((M_new, 4), device="cuda")  # [M_new,4]
            new_rotation[:, 0] = 1.0

            # --------------------------------------------------------
            # 18) 새 Offset 초기화 (k=1이므로 [0,0,0])
            new_offset = torch.zeros((M_new, 3), device="cuda")  # [M_new,3]
            new_offset_scale = new_scale.clone()  # [M_new,3]

            # --------------------------------------------------------
            # 19) 새 extra_level 초기화
            new_extra_level = torch.zeros((M_new,), device="cuda")  # [M_new]

            # --------------------------------------------------------
            # 20) 파라미터 업데이트: split_gaussians 스타일 (_update_param_with_optimizer)
            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                # param은 self.model.<name>의 기존 파라미터 텐서
                if name == "anchor":
                    return torch.nn.Parameter(
                        torch.cat([param, new_anchor]),
                        requires_grad=param.requires_grad,
                    )
                elif name == "scale":
                    return torch.nn.Parameter(
                        torch.cat([param, new_scale]), requires_grad=param.requires_grad
                    )
                elif name == "rotation":
                    return torch.nn.Parameter(
                        torch.cat([param, new_rotation]),
                        requires_grad=param.requires_grad,
                    )
                elif name == "features_albedo":
                    return torch.nn.Parameter(
                        torch.cat([param, new_features_albedo]),
                        requires_grad=param.requires_grad,
                    )
                elif name == "features_specular":
                    return torch.nn.Parameter(
                        torch.cat([param, new_features_specular]),
                        requires_grad=param.requires_grad,
                    )
                elif name == "offset":
                    return torch.nn.Parameter(
                        torch.cat([param, new_offset]),
                        requires_grad=param.requires_grad,
                    )
                elif name == "offset_scale":
                    return torch.nn.Parameter(
                        torch.cat([param, new_offset_scale]),
                        requires_grad=param.requires_grad,
                    )
                elif name == "density":
                    return torch.nn.Parameter(
                        torch.cat([param, new_density]),
                        requires_grad=param.requires_grad,
                    )
                else:
                    # 다른 파라미터는 그대로 반환
                    return param

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                # v는 optimizer.state[p][buffer] (예: exp_avg, exp_avg_sq) 텐서
                pad_shape = (M_new, *v.shape[1:])
                padding = torch.zeros(pad_shape, device=v.device)
                return torch.cat([v, padding], dim=0)

            # 실제 호출: 신규 파라미터(위 update_param_fn으로) 추가하고
            # optimizer 내부 exp_avg/exp_avg_sq까지 확장
            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

            # 21) level, extra_level만 따로 직접 concatenate
            self.model.level = torch.cat(
                [self.model.level, new_level], dim=0
            )  # [N+M_new,1]
            self.model.extra_level = torch.cat(
                [self.model.extra_level, new_extra_level], dim=0
            )  # [N+M_new]
            anchor_grads = torch.cat(
                [anchor_grads, torch.zeros((M_new,), device="cuda")], dim=0
            )

            if self.conf.strategy.print_stats:
                n_before = init_shape
                n_after = self.model.num_gaussians
                logger.info(
                    f"[Level: {cur_level}] Anchor growed {n_before} -> {n_after} ({n_after/n_before*100:.2f}%) gaussians"
                )
        self.reset_densification_buffers()
        if self.conf.strategy.print_stats:
            n_before = init_shape
            n_after = self.model.num_gaussians
            logger.info(
                f"[TOTAL] Anchor growed {n_before} -> {n_after} ({n_after/n_before*100:.2f}%) gaussians"
            )

    def prune_densification_buffers(self, mask):
        super().prune_densification_buffers(mask)
        self.model.level = self.model.level[mask]
        self.model.extra_level = self.model.extra_level[mask]
