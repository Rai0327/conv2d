import torch
import random as ran
import math
from conv import QuantizedConv2dReLU

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
ran.seed(0)

def compute_out_shapes(H, W, kernel_size, stride, padding, dilation):
    H_out = math.floor((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    W_out = math.floor((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return H_out, W_out


def test_1():
    """
    Verify output shapes
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

    conv = QuantizedConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=(ran.random() < 0.5)
    ).cuda()

    x = torch.randn(B, C_in, H, W, device='cuda', requires_grad=False)
    H_out, W_out = compute_out_shapes(H, W, kernel_size, stride, padding, dilation)

    y = conv(x)

    assert y.shape == (B, C_out, H_out, W_out)

    del conv, x, y
    torch.cuda.empty_cache()

for i in range(100):
    test_1()

print("Successfully passed test 1!")
