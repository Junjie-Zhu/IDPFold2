# Docker Usage for IDPFold2

The maintained Docker guide is [`docker/README.md`](docker/README.md). This file is kept for users who find the older root-level Docker README.

The image installs the IDPFold2 environment for training and inference. It does not include checkpoints or pre-download ESM weights; download checkpoints from [Zenodo](https://zenodo.org/records/18239596) and mount them at runtime.

## NVIDIA/CUDA Quickstart

```bash
mkdir -p checkpoints inputs embeddings outputs
docker build -t idpfold2-env .
docker run --rm -it --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/inputs:/workspace/inputs \
  -v $(pwd)/embeddings:/workspace/embeddings \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace/IDPFold-multimer \
  idpfold2-env
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`.

Inside the container, run a small monomer inference:

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

CPU-only containers can be started by removing `--gpus all`, but inference will be much slower.

## Training Smoke Command

```bash
idpfold2-train \
  task_prefix=DOCKER_SMOKE \
  epochs=1 \
  batch_size=1 \
  data.data_dir=/workspace/inputs/TRAIN_DATA_ROOT \
  data.plm_emb_dir=/workspace/embeddings
```

Replace `TRAIN_DATA_ROOT` with prepared training data.

## Ascend 910B Quickstart

Place the CANN installers in the repository root:

- `Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run`
- `Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run`

Build:

```bash
docker build -f Dockerfile.ascend \
  --build-arg CANN_TOOLKIT_RUN="Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run" \
  --build-arg CANN_KERNELS_RUN="Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run" \
  --build-arg TORCH_PACKAGE="torch==2.6.0" \
  --build-arg PYG_PACKAGE="torch-geometric==2.6.1" \
  --build-arg TORCH_NPU_PACKAGE="torch-npu==2.6.0.post3" \
  -t idpfold2-ascend-env .
```

Run:

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

The Ascend image does not install `mmseqs2`; workflows that require clustering need precomputed clusters or a separate `mmseqs2` installation. See [`docker/README.md`](docker/README.md) for full troubleshooting and runtime notes.
