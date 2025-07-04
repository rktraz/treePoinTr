# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-19 17:03:06
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:02:06
# @Email:  cshzxie@gmail.com

import os
import warnings
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Use conda compiler if available
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    os.environ['CC'] = os.path.join(conda_prefix, 'bin', 'gcc')
    os.environ['CXX'] = os.path.join(conda_prefix, 'bin', 'g++')

# Monkey patch the CUDA version check to allow mismatches
import torch.utils.cpp_extension
original_check_cuda_version = torch.utils.cpp_extension._check_cuda_version

def patched_check_cuda_version(compiler_name, compiler_version):
    try:
        original_check_cuda_version(compiler_name, compiler_version)
    except RuntimeError as e:
        if "CUDA version" in str(e) and "mismatches" in str(e):
            warnings.warn(f"CUDA version mismatch detected but proceeding anyway: {e}")
        else:
            raise e

torch.utils.cpp_extension._check_cuda_version = patched_check_cuda_version

# Set architecture for compatibility
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

# Set CUDA paths
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.5')
cuda_include = os.path.join(cuda_home, 'targets/x86_64-linux/include')
cuda_lib = os.path.join(cuda_home, 'targets/x86_64-linux/lib')

setup(name='cubic_feature_sampling',
      version='1.1.0',
      ext_modules=[
          CUDAExtension('cubic_feature_sampling', ['cubic_feature_sampling_cuda.cpp', 'cubic_feature_sampling.cu'],
                        include_dirs=[cuda_include],
                        library_dirs=[cuda_lib],
                        libraries=['cudart'],
                        extra_compile_args={
                            'cxx': ['-std=c++14', '-O3'],
                            'nvcc': ['-std=c++14', '-O3', '--expt-relaxed-constexpr', '--use_fast_math',
                                    f'-I{cuda_include}']
                        })
      ],
      cmdclass={'build_ext': BuildExtension},
      zip_safe=False)
