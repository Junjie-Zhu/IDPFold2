#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="idpfold2",
    version="0.0.1",
    description="Generative conformational ensemble prediction for monomers, multidomain proteins, and protein complexes.",
    author="junjie zhu",
    author_email="shiroyuki@sjtu.edu.cn",
    url="https://github.com/Junjie-Zhu/IDPFold2",
    python_requires=">=3.11",
    install_requires=[
        "biopandas==0.5.1",
        "biopython",
        "biotite==0.41.0",
        "cpdb-protein",
        "dm-tree==0.1.8",
        "einops==0.6",
        "fair-esm",
        "hydra-core==1.3.1",
        "loguru==0.7.2",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "rootutils",
        "scipy",
        "tqdm==4.66.4",
        "wget==3.2",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "idpfold2-train = src.train:main",
            "idpfold2-infer = src.inference:main",
        ]
    },
)
