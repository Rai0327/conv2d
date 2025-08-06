import os
# Adjust environment variables for compilation
os.environ["CC"] = "/usr/bin/gcc-10"
os.environ["CXX"] = "/usr/bin/g++-10"
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'

# import conv2d
from torch.utils.cpp_extension import load
conv2d = load(
    name="conv2d_relu_int8",
    sources=[
        "bindings.cpp",
        "autograd.cpp",
        "conv.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-gencode=arch=compute_61,code=sm_61", # Adjust based on your GPU architecture
    ],
    verbose=False
)

import torch
import torch.nn as nn
    
class QuantizedConv2dReLU(nn.Module):
    """Quantized Conv2d with ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        return conv2d.conv2d_relu_int8(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
    
class TorchConv2dReLU(nn.Module):
    """Baseline PyTorch Conv2d with ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))