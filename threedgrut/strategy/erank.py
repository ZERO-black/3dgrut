import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.gs import GSStrategy

class ERankStrategy(GSStrategy):
    pass
    # def __init__(self, config, model: MixtureOfGaussians) -> None:
    #     super().__init__(config=config, model=model)

    #     self.densify_grad_accum_abs = torch.empty([0, 1])
    #     self.densify_grad_accum_abs_max = torch.empty([0, 1])

    # def get_strategy_parameters(self):
    #     params = super().get_strategy_parameters()
    #     params["densify_grad_accum_abs"] = (self.densify_grad_accum_abs,)
    #     params["densify_grad_accum_abs_max"] = (self.densify_grad_accum_abs_max,)

    #     return params
    
    # def init_densification_buffer(self, checkpoint = None):
    #     super().init_densification_buffer(checkpoint)
    #     if checkpoint is not None:
    #         self.densify_grad_accum_abs = checkpoint["densify_grad_accum_abs"][0].detach()
    #         self.densify_grad_accum_abs_max = checkpoint["densify_grad_accum_abs_max"][0].detach()
    #     else:
    #         num_gaussians = self.model.num_gaussians
    #         self.densify_grad_accum_abs = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
    #         self.densify_grad_accum_abs_max = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)

    # @torch.no_grad()
    # @torch.cuda.nvtx.range("update-gradient-buffer")
    # def update_gradient_buffer(self, sensor_position: torch.Tensor) -> None:
    #     # same as 3dgrut
    #     params_grad = self.model.positions.grad
    #     mask = (params_grad != 0).max(dim=1)[0]
    #     assert params_grad is not None
    #     distance_to_camera = (self.model.positions[mask] - sensor_position).norm(dim=1, keepdim=True)

    #     self.densify_grad_norm_accum[mask] += (
    #         torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2
    #     )
    #     self.densify_grad_norm_denom[mask] += 1
        
    #     # additional gradient / 마스크가 뭔가 다름
    #     self.densify_grad_accum_abs[mask] += (
    #         torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) 
    # )