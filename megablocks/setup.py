#!/usr/bin/env python
import os

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. `torch` is required to install `MegaBlocks`.",) from e

cmdclass = {'build_ext': BuildExtension}
nvcc_flags = ['--ptxas-options=-v', '--optimize=2']

if os.environ.get('TORCH_CUDA_ARCH_LIST'):
    # Let PyTorch builder to choose device to target for.
    device_capability = ''
else:
    device_capability_tuple = torch.cuda.get_device_capability()
    device_capability = f'{device_capability_tuple[0]}{device_capability_tuple[1]}'

if device_capability:
    nvcc_flags.append(f'--generate-code=arch=compute_{device_capability},code=sm_{device_capability}',)

ext_modules = [
    CUDAExtension(
        'megablocks_ops',
        ['csrc/ops.cu'],
        include_dirs=['csrc'],
        extra_compile_args={
            'cxx': ['-fopenmp'],
            'nvcc': nvcc_flags,
        },
    ),
]

setup(
    name="megablocks",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["hydra-core"],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
