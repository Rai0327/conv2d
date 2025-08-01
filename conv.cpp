#include "conv.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void kernel(
    const int8_t* __restrict__ in,
    int8_t* __restrict__ out,
    conv2d& conv
) {
    int pos_out = blockIdx.x * blockDim.x + threadIdx.x; // calculate position in output array

    if (pos_out >= conv.H_out * conv.W_out) {
        return; // out of bounds
    }

    int c_out = blockIdx.y;
    int batch = blockIdx.z;

    int h_out = pos_out / conv.W_out; // calculate output h coordinate
    int w_out = pos_out % conv.W_out; // calculate output w coordinate

    int acc = 0; // accumulator

    for (int c = 0; c < conv.C_in; c++) {
        for (int h = 0; h < conv.k_h; h++) {
            for (int w = 0; w < conv.k_w; w++) {
                int h_in = h_out * conv.stride - conv.padding + h * conv.dilation;
                int w_in = w_out * conv.stride - conv.padding + w * conv.dilation;

                if (h_in < 0 || h_in >= conv.H_in || w_in < 0 || w_in >= conv.W_in) {
                    continue; // skip out-of-bounds input
                }
                
                int in_idx = ((batch * conv.C_in + c) * conv.H_in + h_in) * conv.W_in + w_in;
                int weight_idx = ((c_out * conv.C_in + c) * conv.k_h + h) * conv.k_w + w;

                acc += (((int32_t) in[in_idx]) - conv.x_zp) * ((int32_t) conv.weights[weight_idx] - conv.w_zp);
            }
        }
    }

    float out_acc = acc * conv.x_scale * conv.w_scale + conv.bias[c_out];
    out_acc = relu(out_acc); // apply ReLU activation function

    int acc_quant = std::max(-128, std::min(127, __float2int_rn(out_acc / conv.y_scale) + conv.y_zp));

    int out_idx = ((batch * conv.C_out + c_out) * conv.H_out + h_out) * conv.W_out + w_out;

    out[out_idx] = (int8_t) acc_quant;
}