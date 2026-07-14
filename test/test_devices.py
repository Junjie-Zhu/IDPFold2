import pytest

from conftest import make_tiny_batch


def _run_tiny_inference(device, tiny_model_config):
    torch = pytest.importorskip("torch")
    from src.model.protein_transformer import ProteinTransformerAF3

    model = ProteinTransformerAF3(**tiny_model_config).to(device)
    model.eval()
    batch = make_tiny_batch(torch, device)

    with torch.inference_mode():
        output = model(batch, force_moe_capacity=False)

    assert output["coors_pred"].shape == (1, 4, 3)
    assert output["coors_pred"].device.type == device.type
    assert torch.isfinite(output["coors_pred"]).all()


def test_cpu_inference_smoke(tiny_model_config):
    torch = pytest.importorskip("torch")

    _run_tiny_inference(torch.device("cpu"), tiny_model_config)


@pytest.mark.cuda
def test_gpu_inference_smoke_when_available(tiny_model_config):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")

    _run_tiny_inference(torch.device("cuda:0"), tiny_model_config)
    torch.cuda.synchronize()


@pytest.mark.ascend
def test_ascend_npu_smoke_when_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")

    npu = getattr(torch, "npu", None)
    if npu is None:
        pytest.skip("torch.npu is not available in this environment.")
    if npu.device_count() < 1:
        pytest.skip("No Ascend NPU devices are available in this environment.")

    tensor = torch.ones(4, device="npu:0")
    npu.synchronize()

    assert tensor.device.type == "npu"
    assert tensor.sum().item() == 4
