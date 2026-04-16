# Docker Usage for IDPFold2

This Docker image is an environment container for both training and inference.
It does not download model checkpoints automatically.

## 1) Build image

```bash
docker build -t idpfold2-env .
```

## 2) Prepare host directories

Create local directories before running the container:

- `checkpoints/` for `.pth` files
- `inputs/` for inference CSV or training metadata
- `embeddings/` for PLM embeddings cache
- `outputs/` for logs and generated samples

## 3) Run container (GPU)

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/inputs:/workspace/inputs \
  -v $(pwd)/embeddings:/workspace/embeddings \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace/IDPFold-multimer \
  idpfold2-env
```

## 4) Run container (CPU fallback)

```bash
docker run --rm -it \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/inputs:/workspace/inputs \
  -v $(pwd)/embeddings:/workspace/embeddings \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace/IDPFold-multimer \
  idpfold2-env
```

CPU mode is valid but much slower than GPU.

## 5) Inference example (monomer)

Inside the container shell:

```bash
python src/inference.py \
  prefix=MONOMER_DOCKER \
  ckpt_dir=/workspace/checkpoints/IDPFold2_ema_0.999_260114.pth \
  plm_emb_dir=/workspace/embeddings \
  csv_dir=/workspace/IDPFold-multimer/data/monomer_example.csv \
  nsamples=4 \
  max_batch_length=3500 \
  logging_dir=/workspace/outputs
```

## 6) Training example (smoke run)

Inside the container shell:

```bash
python src/train.py \
  task_prefix=DOCKER_SMOKE \
  epochs=1 \
  batch_size=1 \
  data.data_dir=/workspace/inputs/TRAIN_DATA_ROOT \
  data.plm_emb_dir=/workspace/embeddings
```

Adjust dataset paths and training arguments for your real training job.

## 7) Ascend/CANN image (optional)

If you are using Ascend 910B, use the dedicated `Dockerfile.ascend`.

### Build prerequisites

Place the following two installers in the repository root before building:

- `Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run`
- `Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run`

If your installer filenames differ, override them with build args:

```bash
docker build -f Dockerfile.ascend \
  --build-arg CANN_TOOLKIT_RUN="Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run" \
  --build-arg CANN_KERNELS_RUN="Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run" \
  -t idpfold2-ascend-env .
```

### Build command

```bash
docker build -f Dockerfile.ascend -t idpfold2-ascend-env .
```

### Build with pre-downloaded Miniforge installer (optional)

1. Put Miniforge installer in repository root, for example:
   - `Miniforge3-Linux-aarch64.sh`
2. Build with `MINIFORGE_LOCAL_FILE`:

```bash
docker build -f Dockerfile.ascend \
  --build-arg MINIFORGE_LOCAL_FILE="Miniforge3-Linux-aarch64.sh" \
  -t idpfold2-ascend-env .
```

Notes:

- `Dockerfile.ascend` follows this order:
  1. install Miniforge
  2. install CANN toolkit/kernels
  3. create conda env (`conda create -n idpfold2 python=3.11 pip`)
  4. install base Python requirements with pip inside the conda env
  5. install `pyyaml` + `setuptools`
  6. install torch
  7. install `pyg` with pip (after torch)
  8. install `torch-npu`
- `mmseqs2` is removed from Ascend environment setup.
- You can override torch package at build time:

```bash
docker build -f Dockerfile.ascend \
  --build-arg MINIFORGE_LOCAL_FILE="Miniforge3-Linux-aarch64.sh" \
  --build-arg CANN_TOOLKIT_RUN="Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run" \
  --build-arg CANN_KERNELS_RUN="Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run" \
  --build-arg TORCH_PACKAGE="torch==2.6.0" \
  --build-arg PYG_PACKAGE="pyg==2.6.1" \
  --build-arg TORCH_NPU_PACKAGE="torch-npu==2.6.0.post3" \
  -t idpfold2-ascend-env .
```

- The Ascend installer filenames are currently fixed to the versions above.
- If `MINIFORGE_LOCAL_FILE` is empty or missing, Dockerfile will download Miniforge from `MINIFORGE_URL`.
- CANN installers are executed in non-interactive mode when possible (`--quiet`), and fall back to auto-confirm (`yes Y | ...`) if interactive confirmation is still required.
