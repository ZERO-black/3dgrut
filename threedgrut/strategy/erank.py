import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.gs import GSStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, quaternion_to_so3

class ERankStrategy(GSStrategy):
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        self.densify_grad_accum_abs = torch.empty([0, 1])
        self.densify_grad_accum_abs_max = torch.empty([0, 1])
        self.projection_scale = self.conf.strategy.projection_scale

    def get_strategy_parameters(self):
        params = super().get_strategy_parameters()
        params["densify_grad_accum_abs"] = (self.densify_grad_accum_abs,)
        params["densify_grad_accum_abs_max"] = (self.densify_grad_accum_abs_max,)

        return params
    
    def init_densification_buffer(self, checkpoint = None):
        super().init_densification_buffer(checkpoint)
        if checkpoint is not None:
            self.densify_grad_accum_abs = checkpoint["densify_grad_accum_abs"][0].detach()
            self.densify_grad_accum_abs_max = checkpoint["densify_grad_accum_abs_max"][0].detach()
        else:
            num_gaussians = self.model.num_gaussians
            self.densify_grad_accum_abs = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_grad_accum_abs_max = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)


    def post_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""

        # Update densification buffer:
        if check_step_condition(step, 0, self.conf.strategy.densify.end_iteration, 1):
            with torch.cuda.nvtx.range(f"train_{step}_grad_buffer"):
                self.update_gradient_buffer(batch.T_to_world[0])

        # Clamp density
        if check_step_condition(step, 0, -1, 1) and self.conf.model.density_activation == "none":
            with torch.cuda.nvtx.range(f"train_{step}_clamp_density"):
                self.model.clamp_density()

        return False
        
    @torch.no_grad()
    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(self, T_to_world: torch.Tensor) -> None:
        # change into screen space value
        sensor_position = T_to_world[:3, 3]
        cam_forward_world = T_to_world[:3, 2]
        z = torch.nn.functional.normalize(cam_forward_world, dim=-1)

        params_grad = self.model.positions.grad
        mask = (params_grad != 0).max(dim=1)[0]
        assert params_grad is not None

        g_z_scalar = (params_grad[mask] * z).sum(dim=1, keepdim=True)
        g_z = g_z_scalar * z
        g_xy = params_grad[mask] - g_z

        distance_to_camera = (self.model.positions[mask] - sensor_position).norm(dim=1, keepdim=True)

        self.densify_grad_norm_accum[mask] += (
            torch.norm(g_xy * distance_to_camera, dim=-1, keepdim=True) / self.projection_scale
        )
        z_norm = torch.norm(g_z * distance_to_camera, dim=-1, keepdim=True) / self.projection_scale

        self.densify_grad_accum_abs[mask] += (z_norm)
        self.densify_grad_accum_abs_max[mask] = torch.max(z_norm, self.densify_grad_accum_abs_max[mask])
        self.densify_grad_norm_denom[mask] += 1
    
    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, scene_extent):
        assert (
            self.model.optimizer is not None
        ), "Optimizer need to be initialized before splitting and cloning the Gaussians"
        grads = self.densify_grad_norm_accum / self.densify_grad_norm_denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.densify_grad_accum_abs / self.densify_grad_norm_denom
        grads_abs[grads_abs.isnan()] = 0.0

        grads_norm = torch.norm(grads, dim=-1)
        mask_clone_signed = grads_norm >= self.clone_grad_threshold
        mask_split_signed = grads_norm >= self.split_grad_threshold

        mask_signed_total = mask_clone_signed | mask_split_signed
        ratio = mask_signed_total.float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)


        self.clone_gaussians(grads.squeeze(), grads_abs.squeeze(), Q, scene_extent)
        self.split_gaussians(grads.squeeze(), grads_abs.squeeze(), Q, scene_extent)

        torch.cuda.empty_cache()

    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(self, grads: torch.Tensor, grads_abs: torch.Tensor, grads_abs_threshold, scene_extent: float):
        assert grads is not None, "Positional gradients must be available in order to clone the Gaussians"
        # Extract points that satisfy the gradient condition
        mask = torch.where(grads >= self.clone_grad_threshold, True, False)
        mask_abs = torch.where(grads_abs >= grads_abs_threshold, True, False)

        mask = torch.logical_or(mask, mask_abs)
        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values <= self.relative_size_threshold * scene_extent
        )

        stds = self.model.get_scale()[mask]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask])
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)


        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            if name == "positions":
                p_clone = param[mask] + offsets
            else:
                p_clone = param[mask]
            param_new = torch.cat([param, p_clone])
            return torch.nn.Parameter(param_new, requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat([v, torch.zeros((int(mask.sum()), *v.shape[1:]), device=v.device)])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    @torch.cuda.nvtx.range("split_gaussians")
    def split_gaussians(self, grads: torch.Tensor, grads_abs: torch.Tensor, grads_abs_threshold, scene_extent: float):
        n_init_points = self.model.num_gaussians

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the self.model.positions so only take the points up to size of the initial grad
        padded_grad[: grads.shape[0]] = grads.squeeze()
        padded_grad_abs[: grads_abs.shape[0]] = grads_abs.squeeze()

        mask = torch.where(padded_grad >= self.split_grad_threshold, True, False)
        mask_abs = torch.where(padded_grad_abs >= grads_abs_threshold, True, False)
        mask = torch.logical_or(mask, mask_abs)
        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values > self.relative_size_threshold * scene_extent
        )

        stds = self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Splitted {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [self.split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[mask].repeat(repeats) + offsets  # [2N, 3]
            elif name == "scale":
                p_split = self.model.scale_activation_inv(
                    self.model.scale_activation(param[mask].repeat(repeats)) / (0.8 * self.split_n_gaussians)
                )
            else:
                p_split = param[mask].repeat(repeats)

            p_new = torch.nn.Parameter(torch.cat([param[~mask], p_split]), requires_grad=param.requires_grad)

            return p_new

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros((self.split_n_gaussians * int(mask.sum()), *v.shape[1:]), device=v.device)
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()
    
    def reset_densification_buffers(self) -> None:
        super().reset_densification_buffers()

        self.densify_grad_accum_abs = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.densify_grad_accum_abs.dtype,
        )

        self.densify_grad_accum_abs_max = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.densify_grad_accum_abs_max.dtype,
        )

    def prune_densification_buffers(self, valid_mask: torch.Tensor) -> None:
        super().prune_densification_buffers(valid_mask)
        self.densify_grad_accum_abs = self.densify_grad_accum_abs[valid_mask]
        self.densify_grad_accum_abs_max = self.densify_grad_accum_abs_max[valid_mask]