#pragma once

#include <stdint.h>
#include <torch/types.h>

struct conv2d {
    int C_in, C_out;
    int H_in, H_out;
    int W_in, W_out;
    int k_h, k_w;
    int batch_size, stride, padding, dilation;

    const int8_t* weights;
    const float* bias;

    float x_scale;
    int x_zp;
    const float* w_scale;
    const int* w_zp;

    bool use_relu;
};

void launch_forward_kernel(
    const int8_t* in,
    float* out,
    const conv2d& conv
);

void launch_backward_input_kernel(
    torch::Tensor grad_in, 
    const torch::Tensor grad_out, 
    const torch::Tensor weights,
    const conv2d& conv
);

void launch_backward_weights_kernel(
    const torch::Tensor in,
    const torch::Tensor grad_out,
    torch::Tensor grad_weights, 
    const conv2d& conv
);