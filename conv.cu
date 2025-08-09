#include "conv.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define ERROR_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }


__device__ inline float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void forward_kernel(
    const int8_t* __restrict__ in,
    float* __restrict__ out,
    const conv2d& conv
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
                    continue; // out of bounds
                }
                
                int in_idx = ((batch * conv.C_in + c) * conv.H_in + h_in) * conv.W_in + w_in;
                int weight_idx = ((c_out * conv.C_in + c) * conv.k_h + h) * conv.k_w + w;

                acc += (((int) in[in_idx]) - conv.x_zp) * ((int) conv.weights[weight_idx] - conv.w_zp);
            }
        }
    }

    float out_acc = acc * conv.x_scale * conv.w_scale + (conv.bias ? conv.bias[c_out] : 0.0f);
    int out_idx = ((batch * conv.C_out + c_out) * conv.H_out + h_out) * conv.W_out + w_out;
    out[out_idx] = relu(out_acc);
}

__global__ void backward_input_kernel(
    float* __restrict__ grad_in, 
    const float* __restrict__ grad_out, 
    const int8_t* __restrict__ weights, 
    const conv2d conv
) {
    int w_in = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = blockIdx.y * blockDim.y + threadIdx.y;
    int c_in = blockIdx.z % conv.C_in;
    int batch = blockIdx.z / conv.C_in;

    if (w_in >= conv.W_in || h_in >= conv.H_in || c_in >= conv.C_in) {
        return; // out of bounds
    }

    float acc = 0.0f;

    for (int c = 0; c < conv.C_out; c++) {
        for (int h = 0; h < conv.k_h; h++) {
            for (int w = 0; w < conv.k_w; w++) {
                int h_out = h_in + conv.padding - h * conv.dilation;
                int w_out = w_in + conv.padding - w * conv.dilation;

                if (h_out % conv.stride != 0 || w_out % conv.stride != 0) {
                    continue;
                }

                h_out /= conv.stride;
                w_out /= conv.stride;

                if (h_out < 0 || h_out >= conv.H_out || w_out < 0 || w_out >= conv.W_out) {
                    continue; // out of bounds
                }

                int out_idx = ((batch * conv.C_out + c) * conv.H_out + h_out) * conv.W_out + w_out;
                int weight_idx = ((c * conv.C_in + c_in) * conv.k_h + h) * conv.k_w + w;

                acc += grad_out[out_idx] * (float) (weights[weight_idx] - conv.w_zp) * conv.w_scale;
            }
        }
    }

    int in_idx = ((batch * conv.C_in + c_in) * conv.H_in + h_in) * conv.W_in + w_in;
    grad_in[in_idx] = acc;
}

__global__ void backward_weights_kernel(
    const int8_t* __restrict__ in, 
    const float* __restrict__ grad_out, 
    float* __restrict__ grad_weights, 
    const conv2d conv
) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int c_out = blockIdx.z / conv.C_in;
    int c_in  = blockIdx.z % conv.C_in;

    if (w >= conv.k_w || h >= conv.k_h || c_out >= conv.C_out || c_in >= conv.C_in) {
        return; // out of bounds
    }

    float acc = 0.0f;

    for (int b = 0; b < conv.batch_size; b++) {
        for (int h_out = 0; h_out < conv.H_out; h_out++) {
            for (int w_out = 0; w_out < conv.W_out; w_out++) {
                int h_in = h_out * conv.stride - conv.padding + h * conv.dilation;
                int w_in = w_out * conv.stride - conv.padding + w * conv.dilation;

                if (h_in < 0 || h_in >= conv.H_in || w_in < 0 || w_in >= conv.W_in) {
                    continue; // out of bounds
                }

                int in_idx = ((b * conv.C_in + c_in) * conv.H_in + h_in) * conv.W_in + w_in;
                int out_idx = ((b * conv.C_out + c_out) * conv.H_out + h_out) * conv.W_out + w_out;

                acc += grad_out[out_idx] * (float) (in[in_idx] - conv.x_zp) * conv.x_scale;
            }
        }
    }

    int weight_idx = ((c_out * conv.C_in + c_in) * conv.k_h + h) * conv.k_w + w;
    int total_weights = conv.C_out * conv.C_in * conv.k_h * conv.k_w;
    assert(weight_idx < total_weights);
    grad_weights[weight_idx] = acc;
}

void launch_forward_kernel(
    const int8_t* in,
    float* out,
    const conv2d& conv
) {
    // sizes
    int in_size = conv.C_in * conv.H_in * conv.W_in * conv.batch_size;
    int out_size = conv.C_out * conv.H_out * conv.W_out * conv.batch_size;
    int weights_size = conv.C_out * conv.C_in * conv.k_h * conv.k_w;
    int bias_size = conv.C_out;

    // GPU memory allocation
    int8_t* kernel_in, *kernel_weights;
    float* kernel_out, *kernel_bias;
    conv2d* kernel_conv;

    ERROR_CHECK(cudaMalloc(&kernel_in, in_size * sizeof(int8_t)));
    ERROR_CHECK(cudaMalloc(&kernel_out, out_size * sizeof(float)));
    ERROR_CHECK(cudaMalloc(&kernel_weights, weights_size * sizeof(int8_t)));
    if (conv.bias) {
        ERROR_CHECK(cudaMalloc(&kernel_bias, bias_size * sizeof(float)));
    }
    ERROR_CHECK(cudaMalloc(&kernel_conv, sizeof(conv2d)));

    // Get current CUDA stream
    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    // Copy input, weights, and bias to GPU
    ERROR_CHECK(cudaMemcpyAsync(kernel_in, in, in_size * sizeof(int8_t), cudaMemcpyHostToDevice, curr_stream));
    ERROR_CHECK(cudaMemcpyAsync(kernel_weights, conv.weights, weights_size * sizeof(int8_t), cudaMemcpyHostToDevice, curr_stream));
    if (conv.bias) {
        ERROR_CHECK(cudaMemcpyAsync(kernel_bias, conv.bias, bias_size * sizeof(float), cudaMemcpyHostToDevice, curr_stream));
    } else {
        kernel_bias = nullptr;
    }

    // Copy conv to GPU
    conv2d conv_copy = conv;
    conv_copy.weights = kernel_weights;
    conv_copy.bias = kernel_bias;
    ERROR_CHECK(cudaMemcpyAsync(kernel_conv, &conv_copy, sizeof(conv2d), cudaMemcpyHostToDevice, curr_stream));

    // Launch kernel
    int block_dim = 256; // Number of threads per block
    dim3 blockDim(block_dim);
    dim3 gridDim((conv.H_out * conv.W_out + block_dim - 1) / block_dim, conv.C_out, conv.batch_size); // equivalent to ceil(H_out * W_out / block_dim)
    forward_kernel<<<gridDim, blockDim, 0, curr_stream>>>(kernel_in, kernel_out, *kernel_conv);

    // Copy output
    ERROR_CHECK(cudaMemcpyAsync(out, kernel_out, out_size * sizeof(float), cudaMemcpyDeviceToDevice, curr_stream));

    // Free GPU memory
    ERROR_CHECK(cudaFree(kernel_in));
    ERROR_CHECK(cudaFree(kernel_out));
    ERROR_CHECK(cudaFree(kernel_weights));
    if (conv.bias) {
        ERROR_CHECK(cudaFree(kernel_bias));
    }
    ERROR_CHECK(cudaFree(kernel_conv));

    // Synchronize
    ERROR_CHECK(cudaStreamSynchronize(curr_stream));
}

void launch_backward_input_kernel(
    torch::Tensor grad_in, 
    const torch::Tensor grad_out, 
    const torch::Tensor weights,
    const conv2d& conv
) {
    float* kernel_grad_in = grad_in.data_ptr<float>();
    const float* kernel_grad_out = grad_out.data_ptr<float>();
    const int8_t* kernel_weights = weights.data_ptr<int8_t>();

    // Get current CUDA stream
    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    int block_x = 16;
    int block_y = 16;
    dim3 blockDim(block_x, block_y);
    dim3 gridDim((conv.W_in + block_x - 1) / block_x, (conv.H_in + block_y - 1) / block_y, conv.C_in * conv.batch_size);
    backward_input_kernel<<<gridDim, blockDim, 0, curr_stream>>>(kernel_grad_in, kernel_grad_out, kernel_weights, conv);

    // Check for kernel error
    ERROR_CHECK(cudaGetLastError());
}

void launch_backward_weights_kernel(
    const torch::Tensor in,
    const torch::Tensor grad_out,
    torch::Tensor grad_weights, 
    const conv2d& conv
) {
    const int8_t* kernel_in = in.data_ptr<int8_t>();
    const float* kernel_grad_out = grad_out.data_ptr<float>();
    float* kernel_grad_weights = grad_weights.data_ptr<float>();

    // Get current CUDA stream
    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    int block_x = 16;
    int block_y = 16;
    dim3 blockDim(block_x, block_y);
    dim3 gridDim((conv.k_w + block_x - 1) / block_x, (conv.k_h + block_y - 1) / block_y, conv.C_out * conv.C_in);
    backward_weights_kernel<<<gridDim, blockDim, 0, curr_stream>>>(kernel_in, kernel_grad_out, kernel_grad_weights, conv);

    // Check for kernel error
    ERROR_CHECK(cudaGetLastError());
}