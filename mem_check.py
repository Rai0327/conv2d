import sys, resource
import argparse
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser(description="Evaluate memory usage")
g = parser.add_mutually_exclusive_group(required=True)
g.add_argument("--quant", action="store_true")
g.add_argument("--torch", action="store_true")
args = parser.parse_args()

H, W = 224, 224
B = 8
C_in, C_out = 64, 64
padding = 1
kernel_size = 3
dilation = 1
stride = 1
bias = True

if args.quant:
    from conv import QuantizedConv2dReLU
    conv = QuantizedConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    ).cuda()
elif args.torch:
    from conv import TorchConv2dReLU
    conv = TorchConv2dReLU(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    ).cuda()
else:
    # should be unreachable
    print("Error: Please use one of --quant or --torch to specify which module to run")
    exit(0)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for i in range(100):
    x = torch.randn(B, C_in, H, W, device='cuda', requires_grad=True)
    y = conv(x)
    loss = y.sum()
    loss.backward()

torch.cuda.synchronize()

# Memory report

ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_ram_mb = ru / 1024 if sys.platform != "darwin" else ru / (1024**2)
print("Peak RAM (MB):", f"{peak_ram_mb:.2f}")

print("CUDA peak alloc (MB):", f"{torch.cuda.max_memory_allocated()/1024**2:.2f}")
print("CUDA peak reserved (MB):", f"{torch.cuda.max_memory_reserved()/1024**2:.2f}")
