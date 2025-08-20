import torch
import random as ran
from conv import QuantizedConv2dReLU, TorchConv2dReLU

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
ran.seed(0)


def test_3():
    """
    Verify output correctness with standard PyTorch implementation
    """

    H = ran.randint(8, 256) # height
    W = ran.randint(8, 256) # width
    B = ran.randint(1, 8) # batch size
    C_in = ran.randint(1, 64) # in channels
    C_out = ran.randint(1, 64) # out channels
    padding = ran.randint(0, min(32, H // 2, W // 2)) # padding
    kernel_size = ran.randint(1, min(H, W))
    if kernel_size == 1:
        dilation = 1 # dilation
    else:
        dil_max_h = (H + 2 * padding - 1) // (kernel_size - 1)
        dil_max_w = (W + 2 * padding - 1) // (kernel_size - 1)
        dilation = ran.randint(1, max(1, min(dil_max_h, dil_max_w))) # dilation
    k = dilation * (kernel_size - 1) + 1
    s_max_h = max(1, H + 2 * padding - k + 1)
    s_max_w = max(1, W + 2 * padding - k + 1)
    stride  = ran.randint(1, min(s_max_h, s_max_w)) # stride
    bias = (ran.random() < 0.5)

    quant_conv = QuantizedConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    ).cuda()

    torch_conv = TorchConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
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

    quant_x = torch.randn(B, C_in, H, W, device='cuda', requires_grad=False)
    torch_x = quant_x.detach().clone().requires_grad_(False)

    # Forward pass
    quant_y = quant_conv(quant_x)
    torch_y = torch_conv(torch_x)

    torch.testing.assert_close(quant_y, torch_y, rtol=1e-2, atol=5e-2)

    del quant_conv, torch_conv, quant_x, torch_x, quant_y, torch_y
    torch.cuda.empty_cache()

for i in range(100):
    test_3()

print("Successfully passed test 3!")
