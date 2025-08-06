import torch
import copy
from conv import QuantizedConv2dReLU, TorchConv2dReLU
print("Successfully imported QuantizedConv2dReLU")

# Define test parameters
batch_size = 2
in_channels = 3
out_channels = 4
height, width = 8, 8
kernel_size = 3
stride = 1
padding = 1
dilation = 1

# Instantiate the layer
conv = QuantizedConv2dReLU(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation
).cuda()

torch_conv = TorchConv2dReLU(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation
).cuda()

with torch.no_grad():
    conv.weight.copy_(torch_conv.conv.weight)
    conv.bias.copy_(torch_conv.conv.bias)

# Generate dummy input
x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32, requires_grad=True)
x_torch = copy.deepcopy(x)

# Forward pass
y = conv(x)
y_torch = torch_conv(x_torch)

print("Max abs diff:", (y_torch - y.float()).abs().max().item())
print("Mean abs diff:", (y_torch - y.float()).abs().mean().item())

assert(y.shape == y_torch.shape)  # Should be [batch_size, out_channels, height, width] with same shape due to padding

# Backward pass check
loss = y.sum()
loss.backward()
torch_loss = y_torch.sum()
torch_loss.backward()

# Check gradients
assert(x.grad.shape == x_torch.grad.shape)  # Input gradients should match
assert(conv.weight.grad.shape == torch_conv.conv.weight.grad.shape)  # Weight gradients should match
assert(conv.bias.grad.shape == torch_conv.conv.bias.grad.shape)  # Bias gradients should match

print("Test passed successfully!")