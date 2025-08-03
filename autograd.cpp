#include "conv.h"
#include "autograd.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

torch::Tensor conv2d_relu_int8_forward(
    const torch::Tensor in,
    const torch::Tensor weights,
    const torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(in.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weights.is_cuda(), "Weights must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA");
    TORCH_CHECK(in.dtype() == torch::kInt8, "Expected int8 input");

    int batch_size = in.size(0);
    int C_in = in.size(1);
    int H_in = in.size(2);
    int W_in = in.size(3);
    int C_out = weights.size(0);
    int k_h = weights.size(2);
    int k_w = weights.size(3);
    int H_out = (H_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
    int W_out = (W_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

    torch::Tensor out = torch::empty({batch_size, C_out, H_out, W_out}, torch::dtype(torch::kChar).device(in.device()));

    // Set up conv2d structure
    conv2d conv;
    conv.C_in = C_in;
    conv.C_out = C_out;
    conv.H_in = H_in;
    conv.H_out = H_out;
    conv.W_in = W_in;
    conv.W_out = W_out;
    conv.k_h = k_h;
    conv.k_w = k_w;
    conv.batch_size = batch_size;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;

    conv.x_scale = 1.0f;
    conv.w_scale = 1.0f;
    conv.y_scale = 1.0f;
    conv.x_zp = 0;
    conv.w_zp = 0;
    conv.y_zp = 0;

    conv.weights = weights.data_ptr<int8_t>();
    conv.bias = bias.data_ptr<float>();

    // Launch kernel
    launch_forward_kernel(in.data_ptr<int8_t>(), out.data_ptr<int8_t>(), conv);

    return out;
}

torch::Tensor conv2d_relu_int8_input_backward(
    const torch::Tensor grad_out,
    const torch::Tensor weights,
    int stride, int padding, int dilation
) {
    TORCH_CHECK(grad_out.is_cuda());
    TORCH_CHECK(weights.is_cuda());

    int batch_size = grad_out.size(0);
    int C_out = grad_out.size(1);
    int H_out = grad_out.size(2);
    int W_out = grad_out.size(3);
    int C_in = weights.size(1);
    int k_h = weights.size(2);
    int k_w = weights.size(3);
    int H_in = (H_out - 1) * stride - 2 * padding + dilation * (k_h - 1) + 1;
    int W_in = (W_out - 1) * stride - 2 * padding + dilation * (k_w - 1) + 1;

    torch::Tensor grad_in = torch::empty({batch_size, C_in, H_in, W_in}, torch::dtype(torch::kFloat32).device(grad_out.device()));

    // Set up conv2d structure
    conv2d conv;
    conv.C_in = C_in;
    conv.C_out = C_out;
    conv.H_in = H_in;
    conv.H_out = H_out;
    conv.W_in = W_in;
    conv.W_out = W_out;
    conv.k_h = k_h;
    conv.k_w = k_w;
    conv.batch_size = batch_size;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;

    conv.x_scale = 1.0f;
    conv.w_scale = 1.0f;
    conv.y_scale = 1.0f;
    conv.x_zp = 0;
    conv.w_zp = 0;
    conv.y_zp = 0;

    // Launch kernel
    launch_backward_input_kernel(grad_in, grad_out, weights, conv);

    return grad_in;
}

torch::Tensor conv2d_relu_int8_weights_backward(
    const torch::Tensor in,
    const torch::Tensor grad_out,
    int stride, int padding, int dilation,
    int k_h, int k_w
) {
    TORCH_CHECK(in.is_cuda());
    TORCH_CHECK(grad_out.is_cuda());

    int batch_size = in.size(0);
    int C_in = in.size(1);
    int H_in = in.size(2);
    int W_in = in.size(3);
    int C_out = grad_out.size(1);
    int H_out = (H_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
    int W_out = (W_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

    torch::Tensor grad_weights = torch::empty({C_out, C_in, k_h, k_w}, torch::dtype(torch::kFloat32).device(in.device()));

    // Set up conv2d structure
    conv2d conv;
    conv.C_in = C_in;
    conv.C_out = C_out;
    conv.H_in = H_in;
    conv.H_out = H_out;
    conv.W_in = W_in;
    conv.W_out = W_out;
    conv.k_h = k_h;
    conv.k_w = k_w;
    conv.batch_size = batch_size;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;

    conv.x_scale = 1.0f;
    conv.w_scale = 1.0f;
    conv.y_scale = 1.0f;
    conv.x_zp = 0;
    conv.w_zp = 0;
    conv.y_zp = 0;

    // Launch kernel
    launch_backward_weights_kernel(in, grad_out, grad_weights, conv);

    return grad_weights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Conv2D with ReLU activation for int8 inputs";
    m.def("conv2d_relu_int8_forward", &conv2d_relu_int8_forward);
    m.def("conv2d_relu_int8_input_backward", &conv2d_relu_int8_input_backward);
    m.def("conv2d_relu_int8_weights_backward", &conv2d_relu_int8_weights_backward);
}