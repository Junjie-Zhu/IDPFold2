from __future__ import annotations

import sys
import types

import pandas as pd
import pytest


def _import_generation_dataset(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "rootutils",
        types.SimpleNamespace(setup_root=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.utils.cluster_utils",
        types.SimpleNamespace(log_info=lambda *args, **kwargs: None),
    )
    from src.inference import GenerationDataset

    return GenerationDataset


def test_multimer_esm_embedding_generation_uses_chain_specific_names(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    GenerationDataset = _import_generation_dataset(monkeypatch)

    class FakeAlphabet:
        padding_idx = 0

        def get_batch_converter(self):
            def batch_converter(seq_data):
                labels = [label for label, _sequence in seq_data]
                sequences = [sequence for _label, sequence in seq_data]
                max_len = max(len(sequence) for sequence in sequences) + 2
                tokens = torch.zeros(len(sequences), max_len, dtype=torch.long)
                for i, sequence in enumerate(sequences):
                    tokens[i, : len(sequence) + 2] = 1
                return labels, sequences, tokens

            return batch_converter

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch_tokens, repr_layers, return_contacts):
            del repr_layers, return_contacts
            batch_size, token_count = batch_tokens.shape
            representations = torch.zeros(batch_size, token_count, 2)
            return {"representations": {33: representations}}

    fake_esm = types.SimpleNamespace(
        pretrained=types.SimpleNamespace(
            esm2_t33_650M_UR50D=lambda: (FakeModel(), FakeAlphabet()),
        ),
    )
    monkeypatch.setitem(sys.modules, "esm", fake_esm)

    saved_paths = []

    def fake_save(tensor, path):
        saved_paths.append(path.name if hasattr(path, "name") else path.rsplit("\\", 1)[-1])

    monkeypatch.setattr(torch, "save", fake_save)

    df = pd.DataFrame(
        [
            {
                "test_case": "4mvl",
                "chain_ids": "A:B",
                "sequence": "QDSTSDL:DAEFRH",
            }
        ]
    )

    GenerationDataset.get_esm_embedding(df, tmp_path, load_multimer=True)

    assert saved_paths == ["4mvl_A.pt", "4mvl_B.pt"]


def test_multimer_getitem_loads_chain_specific_embeddings(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    GenerationDataset = _import_generation_dataset(monkeypatch)

    csv_path = tmp_path / "multimer.csv"
    plm_emb_dir = tmp_path / "embeddings"
    plm_emb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "test_case": "4mvl",
                "chain_ids": "A:B",
                "sequence": "ACD:EF",
            }
        ]
    ).to_csv(csv_path, index=False)

    chain_a = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    chain_b = torch.tensor([[4.0, 4.1], [5.0, 5.1]])
    torch.save(chain_a, plm_emb_dir / "4mvl_A.pt")
    torch.save(chain_b, plm_emb_dir / "4mvl_B.pt")

    dataset = GenerationDataset(
        csv_path=str(csv_path),
        plm_emb_dir=str(plm_emb_dir),
        dt=0.25,
        nsamples=3,
        load_multimer=True,
    )

    item = dataset[0]

    assert item["dt"] == 0.25
    assert item["nsamples"] == 3
    assert item["nres"] == 5
    assert item["name"] == "4mvl_A:4mvl_B"
    assert torch.equal(item["plm_emb"], torch.cat([chain_a, chain_b], dim=0))
    assert torch.equal(item["chains"], torch.tensor([1, 1, 1, 2, 2]))
    assert torch.equal(item["residue_idx"], torch.tensor([0, 1, 2, 0, 1]))
    assert item["residue_type"].shape[0] == 5
