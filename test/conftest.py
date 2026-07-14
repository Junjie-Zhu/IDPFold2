from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_model_config():
    return {
        "training": False,
        "token_dim": 16,
        "nlayers": 1,
        "nheads": 4,
        "residual_mha": True,
        "residual_transition": True,
        "parallel_mha_transition": False,
        "use_attn_pair_bias": True,
        "strict_feats": False,
        "feats_init_seq": ["plm_emb", "res_type", "res_idx", "chain_break_per_res"],
        "feats_cond_seq": ["time_emb"],
        "t_emb_dim": 8,
        "idx_emb_dim": 8,
        "dim_cond": 16,
        "plm_in_dim": 8,
        "plm_out_dim": 8,
        "feats_pair_repr": ["xt_pair_dists", "rel_pos"],
        "feats_pair_cond": ["time_emb"],
        "xt_pair_dist_dim": 8,
        "xt_pair_dist_min": 0.1,
        "xt_pair_dist_max": 3.0,
        "r_max": 2,
        "pair_repr_dim": 16,
        "num_registers": 0,
        "use_qkln": True,
        "use_moe": True,
        "n_experts": 2,
        "n_activated_experts": 1,
        "dim_moe_cond": 0,
        "capacity_factor": 1.5,
        "normalize_expert_weights": True,
    }


def make_tiny_batch(torch, device):
    return {
        "x_t": torch.randn(1, 4, 3, device=device),
        "t": torch.full((1,), 0.5, device=device),
        "mask": torch.ones(1, 4, dtype=torch.bool, device=device),
        "residue_type": torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device),
        "plm_emb": torch.randn(1, 4, 8, device=device),
        "chain_break_per_res": torch.zeros(1, 4, device=device),
    }
