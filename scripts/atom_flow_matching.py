from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from src.model.integral import extract_clean_sample, sample_t


def mf_loss(u, u_target, mask, gamma=0.5, c=1e-3):
    natom = mask.sum(dim=-1) * 3  # [*]
    err = (u - u_target.detach()) * mask[..., None]  # [*, n * 14, 3]
    loss = torch.sum(err**2, dim=(-1, -2)) / natom  # [*]

    p = 1.0 - gamma
    w = 1.0 / (loss + c).pow(p)
    return (w.detach() * loss).mean()


def atom_training_predict(
    batch,
    flow_matching,
    model: nn.Module,
    motif_factory: Optional[nn.Module],
    noise_kwargs: dict,
    target_pred: str = 'x_1',
    r_ratio: float = 0.25,
):
    x_0, mask, batch_shape, n, dtype = extract_clean_sample(batch, flow_matching)
    device = x_0.device

    x_0 = flow_matching._mask_and_zero_com(x_0, mask)

    if "mode" in noise_kwargs:
        noise_mode = noise_kwargs["mode"]
        noise_kwargs.pop("mode")
    else:
        noise_mode = "uniform"
    t = sample_t(noise_mode, batch_shape, device, **noise_kwargs)

    assert 0.0 < r_ratio < 1.0, "r_ratio should be in (0, 1)"
    _r = sample_t(noise_mode, batch_shape, device, **noise_kwargs)
    r = torch.minimum(_r, t)
    t = torch.maximum(_r, t)

    _r_mask = r < r_ratio
    r[_r_mask] = t[_r_mask]

    x_1 = flow_matching.sample_reference(
        n=n, shape=batch_shape, device=device, dtype=dtype, mask=mask
    )

    x_t = flow_matching.interpolate(x_0, x_1, t)
    v_t = flow_matching.xt_dot(x_0, x_t, t, mask)

    batch.update({"mask": (mask.view(batch_shape, n // 14, 14).sum(dim=-1) > 0).float(), "atom_mask": mask})
    _model = partial(model, batch_nn=batch)
    jvp_args = (
        lambda x_t, t, r: _model(x_t, t, r),
        (x_t, t, r),
        (v_t, torch.ones_like(t), torch.ones_like(r)),
    )

    u, dudt = torch.func.jvp(*jvp_args)
    u_target = v_t - (t - r) * dudt
    loss = mf_loss(u, u_target, mask)

    return loss


def generating_predict(
    batch,
    flow_matching,
    model: nn.Module,
    noise_kwargs: dict,
):
    return 0
