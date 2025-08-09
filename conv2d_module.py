from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv2d',
    ext_modules=[
        CUDAExtension(
            name='conv2d_relu_int8',
            sources=[
                'bindings.cpp',
                'autograd.cpp',
                'conv.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_61,code=sm_61', # Adjust based on your GPU architecture
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
