#include "autograd.h"
#include <torch/extension.h>

torch::Tensor conv2d_relu_int8_autograd(
    const torch::Tensor& in,
    const torch::Tensor& weights,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int dilation,
    bool use_relu
) {
    return Conv2dReLUInt8Function::apply(in, weights, bias, stride, padding, dilation, use_relu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Conv2D ReLU int8 autograd function";
    m.def("conv2d_relu_int8", &conv2d_relu_int8_autograd);
}