# Docker Usage for IDPFold2

This guide builds an IDPFold2 environment container for inference and training. The images install the code and dependencies, but they do not bundle model checkpoints or pre-download ESM weights.

## Prerequisites

- Docker Engine with enough disk space for the image and conda environment.
- For NVIDIA GPUs: a working NVIDIA driver and NVIDIA Container Toolkit so `docker run --gpus all` works.
- For Ascend 910B: host Ascend drivers, CANN installer `.run` files, and access to the device mounts listed below.
- Download checkpoints from [Zenodo](https://zenodo.org/records/18239596) before inference.

On Windows PowerShell, replace `$(pwd)` in volume mounts with `${PWD}`.

## Host Directories

Create these directories in the repository root or adapt the volume paths:

```bash
mkdir -p checkpoints inputs embeddings outputs
```

- `checkpoints/`: model `.pth` files.
- `inputs/`: custom CSV inputs or training metadata.
- `embeddings/`: PLM embedding cache. ESM weights may download on first use.
- `outputs/`: inference samples and training logs.

## NVIDIA/CUDA Image

Build from the repository root:

```bash
docker build -t idpfold2-env .
```

Run with GPU support:

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/inputs:/workspace/inputs \
  -v $(pwd)/embeddings:/workspace/embeddings \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace/IDPFold-multimer \
  idpfold2-env
```

CPU fallback is valid for small smoke tests, but inference is much slower:

```bash
docker run --rm -it \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/inputs:/workspace/inputs \
  -v $(pwd)/embeddings:/workspace/embeddings \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace/IDPFold-multimer \
  idpfold2-env
```

### Monomer Inference

Inside the container shell:

```bash
idpfold2-infer \
  prefix=MONOMER_DOCKER \
  ckpt_dir=/workspace/checkpoints/IDPFold2_ema_0.999_260114.pth \
  plm_emb_dir=/workspace/embeddings \
  csv_dir=/workspace/IDPFold-multimer/data/monomer_example.csv \
  nsamples=4 \
  max_batch_length=3500 \
  logging_dir=/workspace/outputs
```

The direct script form is equivalent when running from the repository checkout:

```bash
python src/inference.py prefix=MONOMER_DOCKER ckpt_dir=/workspace/checkpoints/IDPFold2_ema_0.999_260114.pth csv_dir=/workspace/IDPFold-multimer/data/monomer_example.csv
```

### Training Smoke Command

Inside the container shell:

```bash
idpfold2-train \
  task_prefix=DOCKER_SMOKE \
  epochs=1 \
  batch_size=1 \
  data.data_dir=/workspace/inputs/TRAIN_DATA_ROOT \
  data.plm_emb_dir=/workspace/embeddings
```

Replace `TRAIN_DATA_ROOT` with a prepared training dataset. The command above checks that the installed CLI and mounted paths resolve; it is not a complete training recipe.

## Ascend/CANN Image

Use `Dockerfile.ascend` for Ascend 910B. This image installs PyTorch, PyG, and `torch-npu` through build arguments so you can match your CANN stack.

### Build Prerequisites

Place the CANN installers in the repository root before building:

- `Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run`
- `Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run`

Optionally place `Miniforge3-Linux-aarch64.sh` in the repository root for offline or mirrored builds.

### Build

```bash
docker build -f Dockerfile.ascend \
  --build-arg MINIFORGE_LOCAL_FILE="Miniforge3-Linux-aarch64.sh" \
  --build-arg CANN_TOOLKIT_RUN="Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run" \
  --build-arg CANN_KERNELS_RUN="Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run" \
  --build-arg TORCH_PACKAGE="torch==2.6.0" \
  --build-arg PYG_PACKAGE="torch-geometric==2.6.1" \
  --build-arg TORCH_NPU_PACKAGE="torch-npu==2.6.0.post3" \
  -t idpfold2-ascend-env .
```

If you do not use a local Miniforge file, omit `MINIFORGE_LOCAL_FILE` and the Dockerfile will download from `MINIFORGE_URL`.

### Run on Ascend 910B

```bash
docker run --rm -it --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /dev:/dev \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/inputs:/workspace/inputs \
  -v $(pwd)/embeddings:/workspace/embeddings \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace/IDPFold-multimer \
  idpfold2-ascend-env
```

Quick check inside the container:

```bash
which npu-smi || true
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__)"
```

If `torch_npu` reports `libhccl.so` or `libascend_hal.so` as missing, run:

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/8.2.RC1/hccl/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/dcmi/lib64:${LD_LIBRARY_PATH}
```

The Ascend image intentionally does not install `mmseqs2`. Inference from precomputed inputs works without it, but training workflows that require on-the-fly clustering need precomputed clusters or a separate `mmseqs2` installation.

## Troubleshooting

- `docker: Error response from daemon: could not select device driver`: install or repair NVIDIA Container Toolkit, then retry `docker run --gpus all`.
- Checkpoint not found: mount the host `checkpoints/` directory and confirm `IDPFold2_ema_0.999_260114.pth` exists inside `/workspace/checkpoints`.
- First inference is slow: ESM weights and embeddings are generated the first time and cached under `/workspace/embeddings`.
- CUDA or NPU out of memory: lower `nsamples` or `max_batch_length`.
- Dependency install failures during build: rebuild after clearing partial layers or pin the accelerator package build arguments to versions supported by your driver/runtime.
