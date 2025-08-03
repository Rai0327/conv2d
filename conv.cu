#include "conv.h"
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define ERROR_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }


void launch_forward_kernel(
    const int8_t* in,
    int8_t* out,
    const conv2d& conv
) {
    // sizes
    int in_size = conv.C_in * conv.H_in * conv.W_in * conv.batch_size;
    int out_size = conv.C_out * conv.H_out * conv.W_out * conv.batch_size;
    int weights_size = conv.C_out * conv.C_in * conv.k_h * conv.k_w;
    int bias_size = conv.C_out;

    // GPU memory allocation
    int8_t* kernel_in, *kernel_out, *kernel_weights;
    float* kernel_bias;
    conv2d* kernel_conv;

    ERROR_CHECK(cudaMalloc(&kernel_in, in_size * sizeof(int8_t)));
    ERROR_CHECK(cudaMalloc(&kernel_out, out_size * sizeof(int8_t)));
    ERROR_CHECK(cudaMalloc(&kernel_weights, weights_size * sizeof(int8_t)));
    ERROR_CHECK(cudaMalloc(&kernel_bias, bias_size * sizeof(float)));
    ERROR_CHECK(cudaMalloc(&kernel_conv, sizeof(conv2d)));

    // Get current CUDA stream
    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();

    // Copy input, weights, and bias to GPU
    ERROR_CHECK(cudaMemcpyAsync(kernel_in, in, in_size * sizeof(int8_t), cudaMemcpyHostToDevice, curr_stream));
    ERROR_CHECK(cudaMemcpyAsync(kernel_weights, conv.weights, weights_size * sizeof(int8_t), cudaMemcpyHostToDevice, curr_stream));
    ERROR_CHECK(cudaMemcpyAsync(kernel_bias, conv.bias, bias_size * sizeof(float), cudaMemcpyHostToDevice, curr_stream));

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
    ERROR_CHECK(cudaMemcpyAsync(out, kernel_out, out_size * sizeof(int8_t), cudaMemcpyDeviceToHost, curr_stream));

    // Free GPU memory
    ERROR_CHECK(cudaFree(kernel_in));
    ERROR_CHECK(cudaFree(kernel_out));
    ERROR_CHECK(cudaFree(kernel_weights));
    ERROR_CHECK(cudaFree(kernel_bias));
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