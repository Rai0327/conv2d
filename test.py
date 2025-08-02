from torch.utils.cpp_extension import load
conv2d = load(name='conv2d', sources=['bindings.cpp', 'conv.cu'])
print("Loaded conv2d extension successfully!")
import torch

x = torch.randint(-128, 127, (1, 3, 32, 32), dtype=torch.int8, device='cuda')
w = torch.randint(-128, 127, (8, 3, 3, 3), dtype=torch.int8, device='cuda')
b = torch.randn(8, device='cuda', dtype=torch.float32)

out = conv2d.conv2d_int8(x, w, b, stride=1, padding=1, dilation=1)
print(out.shape)  # Should be [1, 8, 32, 32]