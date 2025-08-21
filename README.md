# Implementing Custom CUDA Quantized Conv2d + ReLU Module

This repository contains my implementation of a fused int8 quantized 2d convolutional layer and ReLU activation function module along with correctness test cases comparing to the standard PyTorch implementation and an implementation of the VGG16 model architecture with my module. In my module I implement custom forward and backward CUDA kernels with autograd support.

## Requirements

This project was tested with:
- Python 3.13+
- CUDA toolkit 12.9+
- GCC 13+
- SM arch 89+

Run `pip install requirements.txt` to install the necessary python libraries for the test cases and VGG16 model implementation + training.

NOTE: this project can run with lower compatible requirements, make necessary changes to `setup.py` configuration. 

## Build + Install

To build and install the module, run the following:
```
python setup.py build
pip install -e .
```

## Quickstart

After installation, a fused quantized conv2d and ReLU module can be imported from `conv.py` as
```
from conv import QuantizedConv2dReLU
```

Additionally, we provide a `QuantizedConv2d` module without the fused ReLU activation function and these modules' PyTorch counterparts `TorchConv2dReLU` and `TorchConv2d` for comparison.

## Repo Layout
```
tests/                # directory containing test cases
autograd.cpp          # intermediate between autograd.h and cuda.cu to compute paramteres
autograd.h            # implements torch::autograd::Function class for autograd support
bindings.cpp          # create Python bindings for C++ module
conv.cu               # implements forward + backward kernels and their launchers
conv.h                # conv2d struct to store module parameters
conv.py               # Python wrapper classes
quantized_vgg.py      # VGG16 model implementation using quantized conv2d + relu module
requirements.txt      # required python libraries
run_tests.sh          # bash script to run all test cases
setup.py              # build script
torch_vgg.py          # VGG16 model standard PyTorch implementation
train_vgg.py          # training script for VGG16 model
```

## Background

Convolutional neural networks are powerful deep learning models that rely on the 2D convolution operation. This operation slides a learnable filter across the input, performing element-wise multiplication and accumulation on the window to derive the output feature map. As this operation is heavily used in computer vision, it is vastly important to optimize its performance for efficient training and inference. We achieve this with optimized CUDA kernels that leverage powerful NVIDIA GPUs.

Activation functions are functions that modify the outputs of neurons and they are instrumental to neural networks' performance as they introduce non-linearity to the network, allowing it to learn much more complex behaviors. In our module, we fuse the ReLU activation function with our convolutional layer to reduce costly overhead.

Quantization is the process of representing high-precision information in lower-precision formats. In our case, we map inputs and weights, which are usually represented at floating-point accuracy, to lower-bit (int8) integer formats. This allows us to maintain accuracy while reducing computational and memory costs.

## Implementation Details

### Quantization

We perform int8 quantization on the input for our forward pass and store int8 quantized weights. We dequantize to form floating-point outputs and gradients.

Our quantization mapping for an input or weight value $$x \in \mathbb{R}$$ can be represented as the following:

$$q = \text{clamp}(\text{truncate}(\frac{x}{s} + \text{zp}), -128, 127)$$

where $q$ is the quantized output, $s$ is the quantization scale, and $\text{zp}$ is the quantization zero point. We clamp at the -128 and 127 bounds as those are the signed 8-bit integer limits.

Dequantization is achieved by performing the approximate inverse function:

$$x \approx s \cdot (q - \text{zp})$$

We use per-tensor input scales and zero points and per-channel weight scales and zero points. For both input and weight quantization, we use symmetric signed int8 activations, meaning that are scales and zero points for an input tensor $x$ and weight tensor $w$ are computed as:

$$\text{zp}\_{x}, \text{zp}\_{w\_{c}} = 0$$<br>
$$s\_{x} = \frac{\text{max}(|x|)}{127}$$<br>
$$s\_{w\_{c}} = \frac{\text{max}(|w\_{c}|)}{127}$$

where $$\text{zp}\_{x}$$ is the input zero point, $$\text{zp}\_{w\_{c}}$$ is the per-channel weight zero point, $$s\_{x}$$ is the input scale, and $$s\_{w\_{c}}$$ is the per-channel weight scale.

### Forward Pass

The convolution operation for an input tensor $X$, weight tensor $W$, and bias tensor $B$ is defined as:

$$y\_{c\_{\text{out}}, h, w} = \sum\_{c=0}^{C\_{\text{in}} - 1} \sum\_{r=0}^{k\_{h} - 1} \sum\_{s=0}^{k\_{w} - 1} X\_{c, h+r, w+s} \cdot W\_{c\_{\text{out}}, c, r, s} + B\_{c\_{\text{out}}}$$

where $C\_{\text{in}}$ is the number of input channels, $k\_{h}$ is the kernel height, and $k\_{w}$ is the kernel width.

To optimize this logic, when computing an output element our forward kernel uses pointer arithmetic to start at the input and weight rows' base pointers and walk the pointers down the columns according to the dilation parameter. At each point, we subtract input and weight zero points and accumulate in a 32-bit integer. We scale the accumulation once at the end as the computation is mathematically equivalent to dequantizing at each accumulation. Finally, we add the bias and apply the ReLU function.

### Backward Pass

#### Input Gradient

The derivation of the input gradient given a weight $W$ and output gradient w.r.t. output $y$ is:

$$\frac{\partial L}{\partial X_{c, h, w}} = \sum_{c_{\text{out}}=0}^{C_{\text{out}} - 1} \sum_{r=0}^{k_{h} - 1} \sum_{s=0}^{k_{w} - 1} \frac{\partial L}{\partial y_{c_{\text{out}}, \, h-r, \, w-s}} \cdot W_{c_{\text{out}}, c, r, s}$$

where $C\_{\text{out}}$ is the number of output channels, $k\_{h}$ is the kernel height, and $k\_{w}$ is the kernel width.

To optimize this logic, when computing an output element our forward kernel uses pointer arithmetic to start at the output gradient and weight rows' base pointers and advance by stride and dilation. At each point, we subtract input and weight zero points and accumulate in a float. We scale the accumulation once separately for each channel as the computation is mathematically equivalent to dequantizing at each accumulation.

#### Weight Gradient

The derivation of the weight gradient given an input $X$ and output gradient w.r.t. output $y$ is:

$$\frac{\partial L}{\partial W_{c_{\text{out}}, c, r, s}} = \sum_{b=0}^{B-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial y_{c_{\text{out}}, h, w}} \cdot X_{b, c, \, h+r, \, w+s}$$

where $B$ is the batch size, $H_{\text{out}}$ is the output height, and $W_{\text{out}}$ is the output width.

To optimize this logic, when computing an output element our forward kernel uses pointer arithmetic to start at the output gradient and input rows' base pointers and walk the pointers down the columns according to the stride parameter. At each point, we subtract input and weight zero points and accumulate in a float. We scale the accumulation once at the end as the computation is mathematically equivalent to dequantizing at each accumulation.

#### Bias Gradient

The derivation of the bias gradient given an output gradient w.r.t. output $y$ is:

$$\frac{\partial L}{\partial B_{c_{\text{out}}}} = \sum_{b=0}^{B-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial y_{c_{\text{out}}, h, w}}$$

where $B$ is the batch size, $H_{\text{out}}$ is the output height, and $W_{\text{out}}$ is the output width.
