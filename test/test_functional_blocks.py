from __future__ import annotations

import pickle

import pytest

from conftest import make_tiny_batch


def test_checkpoint_loading_round_trip(tmp_path, tiny_model_config):
    torch = pytest.importorskip("torch")
    from src.model.protein_transformer import ProteinTransformerAF3

    source_model = ProteinTransformerAF3(**tiny_model_config)
    checkpoint_path = tmp_path / "tiny_idpfold2_checkpoint.pth"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    target_model = ProteinTransformerAF3(**tiny_model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_result = target_model.load_state_dict(checkpoint["model_state_dict"])

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_cpu_inference_from_loaded_checkpoint(tmp_path, tiny_model_config):
    torch = pytest.importorskip("torch")
    from src.model.protein_transformer import ProteinTransformerAF3

    model = ProteinTransformerAF3(**tiny_model_config)
    checkpoint_path = tmp_path / "tiny_idpfold2_checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    loaded_model = ProteinTransformerAF3(**tiny_model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    batch = make_tiny_batch(torch, torch.device("cpu"))
    with torch.inference_mode():
        output = loaded_model(batch, force_moe_capacity=False)

    assert output["coors_pred"].shape == (1, 4, 3)
    assert torch.isfinite(output["coors_pred"]).all()


def test_quick_analysis_writes_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("biotite")
    from scripts.quick_analysis import main
    from src.utils.pdb_utils import to_pdb_simple

    residue_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    atom_positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    to_pdb_simple(
        atom_positions=atom_positions,
        residue_ids=residue_ids,
        output_dir=str(tmp_path),
        accession_code="tiny",
    )

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["quick_analysis.py", str(tmp_path)]
        main()
    finally:
        sys.argv = old_argv

    metrics_path = tmp_path / "metrics.pkl"
    assert metrics_path.is_file()
    with metrics_path.open("rb") as handle:
        metrics = pickle.load(handle)

    assert metrics["name"] == ["tiny"]
    assert len(metrics["rg_predict"]) == 1
    assert len(metrics["re2e_predict"]) == 1
    assert metrics["re2e_predict"][0].shape == (1,)


def test_moe_router_output_extraction():
    torch = pytest.importorskip("torch")
    from torch import nn
    from src.model.components.moe_modules_torch import MoE

    expert = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8))
    moe = MoE(
        n_experts=3,
        n_activated_experts=2,
        expert=expert,
        dim=8,
        capacity_factor=1.5,
        normalize_expert_weights=True,
        load_balance=False,
    )
    x = torch.randn(2, 5, 8)

    scores, expert_weights, expert_indices = moe.router(x)

    assert scores.shape == (10, 3)
    assert expert_weights.shape == (10, 2)
    assert expert_indices.shape == (10, 2)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(10), atol=1e-6)
    assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(10), atol=1e-6)
    assert expert_indices.min().item() >= 0
    assert expert_indices.max().item() < 3
