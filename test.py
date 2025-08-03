import torch

# Assuming your extension is built and imported as conv2d_relu_int8
from conv import QuantizedConv2dReLU
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

# Generate dummy input
x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32, requires_grad=True)

# Forward pass
y = conv(x)
print("Output shape:", y.shape)  # Should be [batch_size, out_channels, height, width] with same shape due to padding

# Backward pass check
loss = y.sum()
loss.backward()

# Check gradients
print("Gradients on input:", x.grad.shape)
print("Gradients on weight:", conv.weight.grad.shape)
if conv.bias is not None:
    print("Gradients on bias:", conv.bias.grad.shape)
