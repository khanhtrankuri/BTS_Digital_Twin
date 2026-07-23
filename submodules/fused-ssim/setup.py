import os
import sys

if os.name == "nt" and not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
    conda_cuda = os.path.join(os.environ.get("CONDA_PREFIX", sys.prefix), "Library")
    if os.path.isfile(os.path.join(conda_cuda, "bin", "nvcc.exe")):
        os.environ["CUDA_HOME"] = conda_cuda

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

extra_compile_args = {}
cuda_library_dirs = []

if os.name == 'nt':
    extra_compile_args = {"nvcc": ["-Usmall"], "cxx": ["/Usmall"]}
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isfile(os.path.join(cuda_home, "lib", "cudart.lib")):
        cuda_library_dirs.append(os.path.join(cuda_home, "lib"))

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"],
            library_dirs=cuda_library_dirs,
            extra_compile_args=extra_compile_args)
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
