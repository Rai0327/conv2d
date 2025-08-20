import torch
import random as ran
from conv import QuantizedConv2dReLU

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
ran.seed(0)


def test_8():
    """
    Test bias=False parameter by evaluating against zeroed-out bias
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

    bias_conv = QuantizedConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True
    ).cuda()

    conv = QuantizedConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False
    ).cuda()

    with torch.no_grad():
        conv.weight.copy_(bias_conv.weight)
        conv.bias.zero_()

    bias_x = torch.randn(B, C_in, H, W, device='cuda', requires_grad=True)
    x = bias_x.detach().clone().requires_grad_(True)

    # Forward pass
    bias_y = bias_conv(bias_x)
    y = conv(x)

    assert(torch.equal(bias_y, y))

    # Backward pass
    bias_loss = bias_y.sum()
    bias_loss.backward()
    loss = y.sum()
    loss.backward()

    assert(torch.equal(bias_x.grad, x.grad))
    assert(torch.equal(bias_conv.weight.grad, conv.weight.grad))

    del bias_conv, conv, bias_x, x, bias_y, y, bias_loss, loss
    torch.cuda.empty_cache()

for i in range(50):
    test_8()

print("Successfully passed test 8!")
