import torch


def color_normal(normal):
    normal = torch.nn.functional.normalize(normal, dim=-1)
    image = (normal + 1.0) / 2.0
    return image.clamp(0.0, 1.0)


def visualize_depth_t_minmax_rgb(t: torch.Tensor, eps: float = 1e-8):
    """
    t: (B, W, H, 1) ray tracing t 값
    return: (B, W, H, 3) RGB depth (jet colormap)
    """
    # valid t
    valid = torch.isfinite(t) & (t > 0.0)

    B, W, H, _ = t.shape

    t_log = torch.log(t + eps)

    # 배치 전체 기준 min–max
    t_min = torch.quantile(t_log, 0.01)
    t_max = torch.quantile(t_log, 0.99)

    # normalize
    d = (t_log - t_min) / (t_max - t_min + eps)
    d = d.clamp(0.0, 1.0)
    d = torch.where(valid, d, torch.zeros_like(d))

    # jet colormap
    r = torch.clamp(1.5 - torch.abs(4.0 * d - 3.0), 0.0, 1.0)
    g = torch.clamp(1.5 - torch.abs(4.0 * d - 2.0), 0.0, 1.0)
    b = torch.clamp(1.5 - torch.abs(4.0 * d - 1.0), 0.0, 1.0)

    # channel concat
    rgb = torch.cat([r, g, b], dim=-1)
    return rgb
