# import conv2d
from torch.utils.cpp_extension import load
conv2d = load(
    name="conv2d_relu_int8",
    sources=[
        "bindings.cpp",
        "conv.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86"
    ],
    verbose=True
)

import torch
import torch.nn as nn
    
class QuantizedConv2dReLU(nn.Module):
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