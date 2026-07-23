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

os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
]
cxx_flags = []
cuda_library_dirs = []

if os.name == 'nt':
    nvcc_flags.append("-Usmall")
    cxx_flags.append("/Usmall")
    # Conda's cuda-nvcc package places cudart.lib in Library/lib, while
    # torch.utils.cpp_extension only adds Library/lib/x64 by default.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "lib", "cudart.lib")):
        cuda_library_dirs.append(os.path.join(cuda_home, "lib"))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={"nvcc": nvcc_flags, "cxx": cxx_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
