#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

if os.name == "nt" and not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
    conda_cuda = os.path.join(os.environ.get("CONDA_PREFIX", sys.prefix), "Library")
    if os.path.isfile(os.path.join(conda_cuda, "bin", "nvcc.exe")):
        os.environ["CUDA_HOME"] = conda_cuda

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cxx_compiler_flags = []
nvcc_compiler_flags = []
cuda_library_dirs = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    cxx_compiler_flags.append("/Usmall")
    nvcc_compiler_flags.append("-Usmall")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isfile(os.path.join(cuda_home, "lib", "cudart.lib")):
        cuda_library_dirs.append(os.path.join(cuda_home, "lib"))

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
