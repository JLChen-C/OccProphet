from setuptools import find_packages, setup

import os
import torch
from os import path as osp
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        raise EnvironmentError('CUDA is required to compile OccProphet!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='OccProphet',
        version='1.0',
        description=("OccProphet: Pushing Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with Observer-Forecaster-Refiner Framework"),
        author='OccProphet Authors',
        keywords='End-to-end Occupancy Forecasting',
        packages=find_packages(),
        include_package_data=True,
        package_data={'projects.occ_plugin.ops': ['*/*.so']},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="MIT",

        ext_modules=[
            make_cuda_ext(
                name="occ_pool_ext",
                module="projects.occ_plugin.ops.occ_pooling",
                sources=[
                    "src/occ_pool.cpp",
                    "src/occ_pool_cuda.cu",
                ]),
        ],
        cmdclass={'build_ext': BuildExtension})
