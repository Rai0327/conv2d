import torch
from conv import QuantizedConv2dReLU, TorchConv2dReLU

# Define test parameters
batch_size = 5
in_channels = 56
out_channels = 37
height, width = 300, 200
kernel_size = 13
stride = 6
padding = 20
dilation = 5
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

assert (quant_y.shape == torch_y.shape)  # Should be [batch_size, out_channels, height, width]

# Forward equivalence
torch.testing.assert_close(quant_y, torch_y, rtol=1e-2, atol=5e-2)

# Backward pass check
quant_loss = quant_y.sum()
quant_loss.backward()
torch_loss = torch_y.sum()
torch_loss.backward()

# Check gradients
assert (quant_x.grad.shape == torch_x.grad.shape)  # Input gradients should match
assert (quant_conv.weight.grad.shape == torch_conv.conv.weight.grad.shape)  # Weight gradients should match
if bias:
    assert (quant_conv.bias.grad.shape == torch_conv.conv.bias.grad.shape)  # Bias gradients should match

assert (quant_x.grad.isfinite().all().item() and torch_x.grad.isfinite().all().item())  # Check for NaN/Inf in gradients

# Gradient equivalence
torch.testing.assert_close(quant_x.grad, torch_x.grad, rtol=5e-2, atol=5e-1)

# Gradient equivalence for weights
with torch.no_grad():
    # Symmetric per-tensor int8 for activations (same as your C++)
    x = quant_x.detach()
    x_scale = max(x.abs().max().item() / 127.0, 1e-12)
    x_zp = 0
    qX_int = torch.quantize_per_tensor(x, x_scale, x_zp, torch.qint8).int_repr().to(torch.float32)
    x_hat = x_scale * (qX_int - x_zp)  # dequantized int8 activations

# Upstream grad used by your op (post-ReLU mask if you fused ReLU)
if True:  # set False if your op runs without ReLU
    mask = (quant_y > 0).to(torch.float32)
    go_ref = mask  # because loss = sum(y) => upstream is 1 * mask
else:
    go_ref = torch.ones_like(quant_y)

# Compute reference ∂L/∂W with the same conv geometry
w_grad_ref = torch.nn.grad.conv2d_weight(
    x_hat, torch_conv.conv.weight.shape, go_ref,
    stride=stride, padding=padding, dilation=dilation, groups=1
)

# Compare your kernel's weight grad to the reference
torch.testing.assert_close(quant_conv.weight.grad, w_grad_ref, rtol=1e-2, atol=5e-2)

if bias:
    torch.testing.assert_close(quant_conv.bias.grad, torch_conv.conv.bias.grad, rtol=1e-2, atol=1e-2)

print("Test passed successfully!")