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


void launch_kernel(
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
    kernel<<<gridDim, blockDim, 0, curr_stream>>>(kernel_in, kernel_out, *kernel_conv);

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