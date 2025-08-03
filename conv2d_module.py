from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv2d',
    ext_modules=[
        CUDAExtension(
            name='conv2d_relu_int8',
            sources=[
                'bindings.cpp',
                'conv.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
