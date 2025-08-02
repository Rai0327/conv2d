from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = 'conv2d',
    ext_modules = [
        CUDAExtension(
            name = 'conv2d',
            sources = ['bindings.cpp', 'conv.cu']
        )
    ],
    cmdclass = {'build_ext': BuildExtension}
)