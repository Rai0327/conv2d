import torch
import random as ran
from conv import QuantizedConv2dReLU, TorchConv2dReLU

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
ran.seed(0)


def test_5():
    """
    Verify weight grad correctness with ground-truth implementation
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

    quant_x = torch.randn(B, C_in, H, W, device='cuda', requires_grad=True)

    # Forward pass
    quant_y = quant_conv(quant_x)

    # Backward pass
    quant_loss = quant_y.sum()
    quant_loss.backward()

    with torch.no_grad():
        # Symmetric per-tensor int8 for activations
        x = quant_x.detach()
        x_scale = max(x.abs().max().item() / 127.0, 1e-12)
        x_zp = 0
        qX_int = torch.quantize_per_tensor(x, x_scale, x_zp, torch.qint8).int_repr().to(torch.float32)
        x_hat = x_scale * (qX_int - x_zp)  # dequantized int8 activations
    mask = (quant_y > 0).to(torch.float32)
    go_ref = mask

    # Compute reference ∂L/∂W with the same conv geometry
    w_grad_ref = torch.nn.grad.conv2d_weight(
        x_hat, quant_conv.weight.shape, go_ref,
        stride=stride, padding=padding, dilation=dilation, groups=1
    )

    torch.testing.assert_close(quant_conv.weight.grad, w_grad_ref, rtol=5e-1, atol=1.5e-1)

    del quant_conv, w_grad_ref, quant_x, quant_y, quant_loss
    torch.cuda.empty_cache()

for i in range(50):
    test_5()

print("Successfully passed test 5!")
