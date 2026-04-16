FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh

WORKDIR /workspace/IDPFold-multimer
COPY . /workspace/IDPFold-multimer

RUN conda env create -f environment.yaml \
    && conda run -n idpfold2 pip install --no-cache-dir fair-esm \
    && conda run -n idpfold2 pip install --no-cache-dir . \
    && conda clean -a -y

ENV PATH=${CONDA_DIR}/envs/idpfold2/bin:${CONDA_DIR}/bin:${PATH}
ENV CONDA_DEFAULT_ENV=idpfold2

CMD ["/bin/bash"]
