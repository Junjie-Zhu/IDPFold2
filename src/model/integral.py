import random
from functools import partial
from tkinter import X
from typing import Optional, Callable
from math import prod

from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn

import src.model.components.moe_modules as moe_modules
from src.common.atom37_constants import ATOM37_TO_ATOM14_INDICES, RESTYPE_TO_IDX, ATOM14_MASK

nm_to_ang_scale = 10.0
ang_to_nm = lambda trans: trans / nm_to_ang_scale
nm_to_ang = lambda trans: trans * nm_to_ang_scale


def configure_optimizer(model, lr=1e-4):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    return optimizer


def prediction_to_x_clean(nn_out, batch, target_pred='x_1'):
    nn_pred = nn_out["coors_pred"]
    t = batch["t"]  # [*]
    t_ext = t[..., None, None]  # [*, 1, 1]
    x_t = batch["x_t"]  # [*, n, 3]
    if target_pred == "x_1":
        x_1_pred = nn_pred
    elif target_pred == "v":
        x_1_pred = x_t + (1.0 - t_ext) * nn_pred
    else:
        raise IOError(
            f"Wrong parameterization chosen: {target_pred}"
        )
    return x_1_pred


def conditioned_predict(
    batch,
    flow_matching: Callable,
    model: nn.Module,
    model_ag: Optional[nn.Module] = None,
    motif_factory = Optional[nn.Module],
    moe_factory = Optional[nn.Module],
    target_pred = 'x_1',
    guidance_weight = 1.0,
    autoguidance_ratio = 0.0,
    motif_conditioning=False,
    moe_conditioning=False,
):
    if motif_conditioning:
        assert ("fixed_structure_mask" not in batch or "x_motif" not in batch), \
            "Motif conditioning is not supported with fixed structure mask or x_motif in batch."
        batch.update(motif_factory(batch, zeroes=True))

    if moe_conditioning:
        batch.update(moe_factory(batch, zeroes=True))

    nn_out = model(batch)
    x_pred = prediction_to_x_clean(nn_out, batch, target_pred=target_pred)

    if guidance_weight != 1.0:
        assert 0.0 <= autoguidance_ratio <= 1.0
        if autoguidance_ratio > 0.0:  # Use auto-guidance
            assert model_ag is not None, "Model for auto-guidance must be provided"
            nn_out_ag = model_ag(batch)
            x_pred_ag = prediction_to_x_clean(nn_out_ag, batch, target_pred=target_pred)
        else:
            x_pred_ag = torch.zeros_like(x_pred)

        if autoguidance_ratio < 1.0:  # Use CFG
            assert (
                    "plm_embedding" in batch
            ), "Only support CFG when sequence embedding is provided"
            uncond_batch = batch.copy()
            uncond_batch.pop("plm_embedding")
            nn_out_uncond = model(uncond_batch)
            x_pred_uncond = prediction_to_x_clean(nn_out_uncond, uncond_batch, target_pred=target_pred)
        else:
            x_pred_uncond = torch.zeros_like(x_pred)

        x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (
            autoguidance_ratio * x_pred_ag + (1 - autoguidance_ratio) * x_pred_uncond
        )

    v = flow_matching.xt_dot(x_pred, batch["x_t"], batch["t"], batch["mask"])
    return x_pred, v


def sample_t(mode, shape, device, **kwargs):
    if mode == "uniform":
        t_max = kwargs["p2"]
        return torch.rand(shape, device=device) * t_max  # [*]
    elif mode == "logit-normal":
        mean = kwargs["p1"]
        std = kwargs["p2"]
        noise = torch.randn(shape, device=device) * std + mean  # [*]
        return torch.nn.functional.sigmoid(noise)  # [*]
    elif mode == "beta":
        p1 = kwargs["p1"]
        p2 = kwargs["p2"]
        dist = torch.distributions.beta.Beta(p1, p2)
        return dist.sample(shape).to(device)
    elif mode == "mix_up02_beta":
        p1 = kwargs["p1"]
        p2 = kwargs["p2"]
        dist = torch.distributions.beta.Beta(p1, p2)
        samples_beta = dist.sample(shape).to(device)
        samples_uniform = torch.rand(shape, device=device)
        u = torch.rand(shape, device=device)
        return torch.where(u < 0.02, samples_uniform, samples_beta)
    else:
        raise NotImplementedError(
            f"Sampling mode for t {mode} not implemented"
        )


def apply_random_rotation(x, mask, flow_matching):
    assert (
            x.ndim == 3
    ), f"Augmetations can only be used for simple (x_1) batches [b, n, 3], current shape is {x.shape}"
    assert (
            mask.ndim == 2
    ), f"Augmetations can only be used for simple (mask) batches [b, n], current shape is {mask.shape}"

    # Sample and apply rotations
    rots = sample_uniform_rotation(
        shape=x.shape[:-2], dtype=x.dtype, device=x.device
    )  # [naug * b, 3, 3]
    x_rot = torch.matmul(x, rots)
    return flow_matching._mask_and_zero_com(x_rot, mask), mask


def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> torch.Tensor:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


def extract_clean_sample(batch, flow_matching, global_rotation=True):
    x_1 = batch["coords"][:,:,1,:]  # [b, n, 3]
    mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
    if global_rotation:
        x_1, mask = apply_random_rotation(x_1, mask, flow_matching)
    batch_shape = x_1.shape[:-2]
    n = x_1.shape[-2]
    return (
        ang_to_nm(x_1),
        mask,
        batch_shape,
        n,
        x_1.dtype,
    )


def extract_atom_clean_sample(batch, flow_matching, global_rotation=True):
    coords_atom37 = batch["coords"]  # [b, n, 37, 3]
    mask_atom37 = batch["mask_dict"]["coords"]  # [b, n, 37, 1]
    restype = batch["residues"]  # [b, n]

    b, nres, _, coord_shape = coords_atom37.shape

    x_1, mask = atom37_to_atom14_tensor(coords_atom37, mask_atom37, restype)
    x_1 = x_1.view(b, nres * 14, coord_shape)
    mask = mask.view(b, nres * 14)
    if global_rotation:
        x_1, mask = apply_random_rotation(x_1, mask, flow_matching)
    return (
        ang_to_nm(x_1),
        mask,
        (b,),
        nres * 14,
        x_1.dtype,
    )


def atom37_to_atom14_tensor(x, atom37_mask, restype):
    """
    Convert atom37 representation to atom14 representation using Torch.

    Args:
        x: Atom37 coordinates, shape [B, N, 37, D].
        atom37_mask: Atom37 presence mask, shape [B, N, 37].
        restype: Residue types, shape [B, N].

    Returns:
        A tuple containing:
            - atom14_tensor: Atom14 coordinates, shape [B, N, 14, D].
            - atom14_presence_mask: A boolean mask for the atom14 tensor,
              shape [B, N, 14]. This is a combination of the residue type mask
              and the input atom37_mask.
    """
    b, nres, _, d = x.shape
    device = x.device

    # Vectorize the residue type lookup using a mapping from string to index.
    residue_indices_list = [[RESTYPE_TO_IDX[rt] for rt in batch] for batch in restype]
    residue_indices = torch.tensor(residue_indices_list, dtype=torch.long, device=device)  # [B, N]

    # Gather the atom37 indices and mask for the given residue types
    atom37_indices = ATOM37_TO_ATOM14_INDICES.to(device)[residue_indices]  # [B, N, 14]
    type_mask = ATOM14_MASK.to(device)[residue_indices]  # [B, N, 14]

    # Use advanced indexing to gather data from the atom37 tensor
    # The 'torch.arange' creates a broadcastable array for the batch and residue dimensions.
    batch_idx = torch.arange(b, device=device)[:, None, None]
    res_idx = torch.arange(nres, device=device)[None, :, None]

    atom14_tensor = x[
        batch_idx,
        res_idx,
        atom37_indices,
        :
    ]

    gathered_mask = atom37_mask[
        batch_idx,
        res_idx,
        atom37_indices
    ]

    # Combine the two masks: an atom is present only if it's in the residue type
    # mask AND the input atom37_mask.
    atom14_presence_mask = gathered_mask & type_mask

    # Zero out the coordinates for non-existent atoms using the combined mask.
    atom14_tensor = atom14_tensor * atom14_presence_mask.unsqueeze(-1)

    return atom14_tensor, atom14_presence_mask


def compute_fm_loss(
    x_1: torch.Tensor,
    x_1_pred: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes and logs flow matching loss.

    Args:
        x_1: True clean sample, shape [*, n, 3].
        x_1_pred: Predicted clean sample, shape [*, n, 3].
        t: Interpolation time, shape [*].
        mask: Boolean residue mask, shape [*, nres].

    Returns:
        Flow matching loss.
    """
    nres = torch.sum(mask, dim=-1) * 3  # [*]

    err = (x_1 - x_1_pred) * mask[..., None]  # [*, n, 3]
    loss = torch.sum(err**2, dim=(-1, -2)) / nres  # [*]

    total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

    loss = loss * total_loss_w  # [*]
    return loss


def compute_bond_loss(
    x_1: torch.Tensor,
    x_1_pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes and logs bond loss.

    Args:
        x_1: True clean sample, shape [*, n, 3].
        x_1_pred: Predicted clean sample, shape [*, n, 3].
        mask: Boolean residue mask, shape [*, nres].

    Returns:
        Bond loss.
    """
    pair_mask = mask[..., None] * mask[..., None, :]  # [*, n, n]

    x_1_dist = torch.cdist(x_1, x_1)
    x_1_pred_dist = torch.cdist(x_1_pred, x_1_pred)

    distance_mask = (x_1_dist < 10.0).float()
    pair_mask = pair_mask * distance_mask

    err = (x_1_dist - x_1_pred_dist) ** 2  # [*, n, n, 3]
    loss = torch.sum(err * pair_mask, dim=(-1, -2)) / torch.sum(pair_mask, dim=(-1, -2))  # [*]
    return loss


def compute_moe_loss(weight, num_layers, num_experts, top_k):
    batched_loss = moe_modules.batched_load_balancing_loss(weight, num_layers, num_experts, top_k)
    moe_modules.clear_load_balancing_loss()
    return batched_loss


def training_predict(
    batch,
    flow_matching,
    model: nn.Module,
    motif_factory: Optional[nn.Module],
    moe_factory: Optional[nn.Module],
    noise_kwargs: dict,
    target_pred: str = 'x_1',
    motif_conditioning=False,
    moe_conditioning=False,
    self_conditioning=False,
    moe_loss_weight=0.0
):
    # get input data
    x_1, mask, batch_shape, n, dtype = extract_clean_sample(batch, flow_matching)
    device = x_1.device

    # Center and mask input
    x_1 = flow_matching._mask_and_zero_com(x_1, mask)

    # sample noise scale
    if "mode" in noise_kwargs:
        noise_mode = noise_kwargs["mode"]
        noise_kwargs.pop("mode")
    else:
        noise_mode = "uniform"
    t = sample_t(noise_mode, batch_shape, device, **noise_kwargs)
    x_0 = flow_matching.sample_reference(
        n=n, shape=batch_shape, device=device, dtype=dtype, mask=mask
    )

    if motif_conditioning:
        batch.update(motif_factory(batch))
        x_1 = batch["x_1"]
    
    if moe_conditioning:
        batch.update(moe_factory(batch))

    # interpolation
    x_t = flow_matching.interpolate(x_0, x_1, t)

    batch.update({
        "x_t": x_t,
        "t": t,
        "mask": mask,
    })

    # self-conditioning
    if self_conditioning and random.random() < 0.5:
        x_sc = prediction_to_x_clean(model(batch), batch, target_pred=target_pred)
        batch['x_sc'] = x_sc

    # model prediction
    nn_out = model(batch)
    x_pred = prediction_to_x_clean(nn_out, batch, target_pred=target_pred)

    # loss
    fm_loss = compute_fm_loss(x_1, x_pred, t, mask)
    fm_loss = torch.mean(fm_loss)
    if moe_loss_weight != 0.0:
        try:
            n_layers = model.nlayers
            n_experts = model.n_experts
            top_k = model.top_k
        except:
            n_layers = model.module.nlayers
            n_experts = model.module.n_experts
            top_k = model.module.top_k
        moe_loss = compute_moe_loss(
            weight=moe_loss_weight,
            num_layers=n_layers,
            num_experts=n_experts,
            top_k=top_k,
        )
    else:
        moe_loss = 0.0

    loss_dict = {
        "fm_loss": fm_loss.item(),
        "moe_loss": moe_loss.item(),
    }
    return fm_loss + moe_loss, loss_dict


def generating_predict(
    batch,
    flow_matching: Callable,
    model: nn.Module,
    model_ag: Optional[nn.Module] = None,
    motif_factory: Optional[nn.Module] = None,
    moe_factory: Optional[nn.Module] = None,
    target_pred: str = 'x_1',
    guidance_weight = 1.0,
    autoguidance_ratio = 0.0,
    schedule_args: dict = None,
    sampling_args: dict = None,
    motif_conditioning = False,
    moe_conditioning = False,
    self_conditioning = False,
    device = 'cpu'
):
    cleaned_conditioned_predict = partial(
        conditioned_predict,
        flow_matching=flow_matching,
        model=model,
        model_ag=model_ag,
        motif_factory=motif_factory,
        moe_factory=moe_factory,
        target_pred=target_pred,
        guidance_weight=guidance_weight,
        autoguidance_ratio=autoguidance_ratio,
        motif_conditioning=motif_conditioning,
        moe_conditioning=moe_conditioning,
    )

    nsamples = batch["nsamples"].item() if isinstance(batch["nsamples"], torch.Tensor) else batch["nsamples"]
    nres = batch["nres"]
    mask = batch["mask"] if 'mask' in batch else torch.ones(nsamples, nres).long().bool().to(device)
    plm_embedding = batch["plm_emb"] if "plm_emb" in batch else None
    residue_type = batch["residue_type"] if "residue_type" in batch else None
    residue_idx = batch["residue_idx"] if "residue_idx" in batch else None
    chains = batch["chains"] if "chains" in batch else None

    if len(mask.shape) == 1:
        mask = mask.unsqueeze(0).repeat(nsamples, 1)
        plm_embedding = plm_embedding.unsqueeze(0)
        residue_type = residue_type.unsqueeze(0)
        residue_idx = residue_idx.unsqueeze(0)
        chains = chains.unsqueeze(0)

    plm_embedding = plm_embedding.repeat(nsamples, 1, 1)
    residue_type = residue_type.repeat(nsamples, 1)
    residue_idx = residue_idx.repeat(nsamples, 1)
    chains = chains.repeat(nsamples, 1)

    pred_structure = flow_matching.full_simulation(
        cleaned_conditioned_predict,
        dt=batch["dt"].to(dtype=torch.float32),
        nsamples=nsamples,
        n=nres,
        self_cond=self_conditioning,
        plm_embedding=plm_embedding,
        residue_type=residue_type,
        residue_idx=residue_idx,
        chains=chains,
        device=device,
        mask=mask,
        dtype=torch.float32,
        schedule_mode=schedule_args.get('schedule_mode', 'log'),
        schedule_p=schedule_args.get('schedule_p', 2.0),
        sampling_mode=sampling_args["sampling_mode"],
        sc_scale_noise=sampling_args["sc_scale_noise"],
        sc_scale_score=sampling_args["sc_scale_score"],
        gt_mode=sampling_args["gt_mode"],
        gt_p=sampling_args["gt_p"],
        gt_clamp_val=sampling_args["gt_clamp_val"],
        x_motif=None,  # not implemented yet
        fixed_sequence_mask=None,  # not implemented yet
        fixed_structure_mask=None,  # not implemented yet
    )

    moe_modules.clear_load_balancing_loss()
    return pred_structure


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
    noise_kwargs: dict,
    r_ratio: float = 0.25,
):
    x_0, mask, batch_shape, n, dtype = extract_atom_clean_sample(batch, flow_matching)  # b, n * 14, 3
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
    t = torch.maximum(_r, t)
    r = torch.minimum(_r, t)

    _r_mask = r < r_ratio
    r[_r_mask] = t[_r_mask]

    x_1 = flow_matching.sample_reference(
        n=n, shape=batch_shape, device=device, dtype=dtype, mask=mask
    )

    x_t = flow_matching.interpolate(x_1, x_0, t) * mask[..., None]
    v_t = flow_matching.xt_dot(x_0, x_t, t, mask) * mask[..., None]

    x_t = x_t.view(batch_shape, n, 14 * 3)

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


def atom_generating_predict(
    batch,
    model: nn.Module,
    flow_matching: Callable,
    n_step: int = 5,
    device = 'cpu'
):
    nsamples = batch["nsamples"]
    nres = batch["nres"]
    mask = batch["mask"].squeeze(0) if 'mask' in batch else torch.ones(nsamples, nres).long().bool().to(device)

    z = flow_matching.sample_reference(
        n=nres, shape=(nsamples,), device=device, dtype=torch.float32, mask=mask
    )

    t_vals = torch.linspace(0.0, 1.0, n_step + 1, device=device)  # [n_step + 1]

    for i in range(n_step):
        r = torch.full((nsamples,), t_vals[i], device=device)
        t = torch.full((nsamples,), t_vals[i + 1], device=device)

        v_t = model(z, t, r, batch_nn=batch)  # [nsamples, n * 14, 3]
        z = z + (t - r) * v_t

    return z
