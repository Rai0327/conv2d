import torch
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
bias = True

# Instantiate the layer
quant_conv = QuantizedConv2dReLU(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    bias=bias
).cuda()

torch_conv = TorchConv2dReLU(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    bias=bias
).cuda()

with torch.no_grad():
    quant_conv.weight.copy_(torch_conv.conv.weight)
    if bias:
        quant_conv.bias.copy_(torch_conv.conv.bias)

# Generate dummy input
quant_x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32, requires_grad=True)
torch_x = quant_x.detach().clone().requires_grad_(True)

# Zero grads
quant_conv.zero_grad(set_to_none=True)
torch_conv.zero_grad(set_to_none=True)

# Forward pass
quant_y = quant_conv(quant_x)
torch_y = torch_conv(torch_x)

diff = torch_y - quant_y
print("Max abs diff:", diff.abs().max().item())
print("Mean abs diff:", diff.abs().mean().item())
print("Median abs diff:", diff.abs().median().item())
assert(diff.abs().mean().item() < 0.01 and diff.abs().median().item() < 0.01) # Acceptable difference threshold

assert(quant_y.shape == torch_y.shape)  # Should be [batch_size, out_channels, height, width]

# Backward pass check
quant_loss = quant_y.sum()
quant_loss.backward()
torch_loss = torch_y.sum()
torch_loss.backward()

# Check gradients
assert(quant_x.grad.shape == torch_x.grad.shape)  # Input gradients should match
assert(quant_conv.weight.grad.shape == torch_conv.conv.weight.grad.shape)  # Weight gradients should match
if bias:
    assert(quant_conv.bias.grad.shape == torch_conv.conv.bias.grad.shape)  # Bias gradients should match

print("Test passed successfully!")