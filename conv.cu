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

inline __host__ __device__ long long ceil_div_ll(long long a, long long b) {
    return (a + b - 1) / b;
}

__host__ __device__ __forceinline__ int div_floor(int a, int b) {
  int q = a / b, r = a % b;
  if (r != 0 && ((r > 0) != (b > 0))) --q;
  return q;
}
__host__ __device__ __forceinline__ int div_ceil(int a, int b) {
  int q = a / b, r = a % b;
  if (r != 0 && ((r > 0) == (b > 0))) ++q;
  return q;
}

constexpr int BLOCK = 256;

__global__ void forward_kernel(
    const int8_t* __restrict__ in,
    float* __restrict__ out,
    const conv2d conv
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = idx % conv.W_out;
    idx /= conv.W_out;
    int h_out = idx % conv.H_out;
    idx /= conv.H_out;
    int c_out = idx % conv.C_out;
    int batch = idx / conv.C_out;

    int kh_min = max(0, (conv.padding - h_out * conv.stride + conv.dilation - 1) / conv.dilation);
    int kh_max = min(conv.k_h, (conv.padding + conv.H_in - 1 - h_out * conv.stride) / conv.dilation + 1);
    int kw_min = max(0, (conv.padding - w_out * conv.stride + conv.dilation - 1) / conv.dilation);
    int kw_max = min(conv.k_w, (conv.padding + conv.W_in - 1 - w_out * conv.stride) / conv.dilation + 1);

    int w_zp = conv.w_zp[c_out];

    int acc = 0; // accumulator

    for (int c = 0; c < conv.C_in; ++c) {
        for (int h = kh_min; h < kh_max; ++h) {
            int h_in = h_out * conv.stride - conv.padding + h * conv.dilation;
            const int8_t* x_row = in + ((batch * conv.C_in + c) * conv.H_in + h_in) * conv.W_in;
            const int8_t* w_row = conv.weights + ((c_out * conv.C_in + c) * conv.k_h + h) * conv.k_w;
            const int8_t* in_ptr = x_row + w_out * conv.stride - conv.padding + kw_min * conv.dilation;

            #pragma unroll
            for (int w = kw_min; w < kw_max; ++w) {
                acc += (int(*in_ptr) - conv.x_zp) * (int(w_row[w]) - w_zp);
                in_ptr += conv.dilation;
            }
        }
    }

    float out_acc = acc * conv.x_scale * conv.w_scale[c_out] + (conv.bias ? conv.bias[c_out] : 0.0f);
    int out_idx = ((batch * conv.C_out + c_out) * conv.H_out + h_out) * conv.W_out + w_out;
    out[out_idx] = conv.use_relu ? relu(out_acc) : out_acc;
}

__global__ void backward_input_kernel(
    float* __restrict__ grad_in, 
    const float* __restrict__ grad_out, 
    const int8_t* __restrict__ weights, 
    const conv2d conv
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_in = idx % conv.W_in;
    idx /= conv.W_in;
    int h_in = idx % conv.H_in;
    idx /= conv.H_in;
    int c_in = idx % conv.C_in;
    int batch = idx / conv.C_in;

    const int kh_min = max(0, div_ceil(h_in + conv.padding - (conv.H_out - 1) * conv.stride, conv.dilation));
    const int kh_max = min(conv.k_h - 1, div_floor(h_in + conv.padding, conv.dilation));
    const int kw_min = max(0, div_ceil(w_in + conv.padding - (conv.W_out - 1) * conv.stride, conv.dilation));
    const int kw_max = min(conv.k_w - 1, div_floor(w_in + conv.padding, conv.dilation));

    float acc = 0.0f;

    for (int c = 0; c < conv.C_out; ++c) {
        const float*  __restrict__ grad_out_base = grad_out + ((batch * conv.C_out + c) * conv.H_out) * conv.W_out;
        const int8_t* __restrict__ weights_base = weights + ((c * conv.C_in + c_in) * conv.k_h) * conv.k_w;
        int w_zp = conv.w_zp[c];
        float w_scale = conv.w_scale[c];
        for (int h = kh_min; h <= kh_max; ++h) {
            int h_out = h_in + conv.padding - h * conv.dilation;
            if (h_out % conv.stride) {
                continue; // skip if not aligned with stride
            }
            h_out /= conv.stride;
            if (h_out < 0 || h_out >= conv.H_out) {
                continue; // out of bounds
            }

            const float*  __restrict__ grad_out_row = grad_out_base + h_out * conv.W_out;
            const int8_t* __restrict__ weights_row  = weights_base + h * conv.k_w;

            #pragma unroll 1
            for (int w = kw_min; w <= kw_max; ++w) {
                int w_out = w_in + conv.padding - w * conv.dilation;
                if (w_out % conv.stride) {
                    continue; // skip if not aligned with stride
                }
                w_out /= conv.stride;
                if (w_out < 0 || w_out >= conv.W_out) {
                    continue; // out of bounds
                }

                acc += grad_out_row[w_out] * (float)(weights_row[w] - w_zp) * w_scale;
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
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w = idx % conv.k_w;
    idx /= conv.k_w;
    int h = idx % conv.k_h;
    idx /= conv.k_h;
    int c_out = idx % conv.C_out;
    int c_in = idx / conv.C_out;

    if (w >= conv.k_w || h >= conv.k_h || c_out >= conv.C_out || c_in >= conv.C_in) {
        return; // out of bounds
    }

    const int kh_diff = h * conv.dilation - conv.padding;
    const int kw_diff = w * conv.dilation - conv.padding;
    int h_out_min = max(0, div_ceil(-kh_diff, conv.stride));
    int h_out_max = min(conv.H_out - 1, div_floor(conv.H_in - 1 - kh_diff, conv.stride));
    int w_out_min = max(0, div_ceil(-kw_diff, conv.stride));
    int w_out_max = min(conv.W_out - 1, div_floor(conv.W_in - 1 - kw_diff, conv.stride));

    float acc = 0.0f;

    for (int b = 0; b < conv.batch_size; ++b) {
        const float*  __restrict__ grad_out_base = grad_out + ((b * conv.C_out + c_out) * conv.H_out) * conv.W_out;
        const int8_t* __restrict__ in_base = in + ((b * conv.C_in + c_in) * conv.H_in) * conv.W_in;
        for (int h_out = h_out_min; h_out <= h_out_max; ++h_out) {
            const int h_in = h_out * conv.stride + kh_diff;
            const float*  __restrict__ grad_out_row = grad_out_base + h_out * conv.W_out + w_out_min;
            const int8_t* __restrict__ in_row  = in_base + h_in  * conv.W_in + (w_out_min * conv.stride + kw_diff);

            #pragma unroll 4
            for (int w_out = w_out_min; w_out <= w_out_max; ++w_out) {
                acc += (*(grad_out_row++)) * ((float) ((int) (*in_row) - conv.x_zp)) * conv.x_scale;
                in_row += conv.stride;
            }
        }
    }

    int weight_idx = ((c_out * conv.C_in + c_in) * conv.k_h + h) * conv.k_w + w;
    grad_weights[weight_idx] = acc;
}

void launch_forward_kernel(
    const int8_t* in,
    float* out,
    const conv2d& conv
) {
    // Get current CUDA stream
    cudaStream_t curr_stream = at::cuda::getCurrentCUDAStream();
    
    long long N = 1LL * conv.batch_size * conv.C_out * conv.H_out * conv.W_out;
    dim3 blockDim(BLOCK, 1, 1);
    dim3 gridDim(ceil_div_ll(N, BLOCK), 1, 1);

    forward_kernel<<<gridDim, blockDim, 0, curr_stream>>>(in, out, conv);

    // Check for kernel error
    ERROR_CHECK(cudaGetLastError());
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
    long long N = 1LL * conv.batch_size * conv.C_in * conv.H_in * conv.W_in;
    dim3 blockDim(BLOCK, 1, 1);
    dim3 gridDim(ceil_div_ll(N, BLOCK), 1, 1);

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
    long long N = 1LL * conv.C_out * conv.C_in * conv.k_h * conv.k_w;
    dim3 blockDim(BLOCK, 1, 1);
    dim3 gridDim(ceil_div_ll(N, BLOCK), 1, 1);
    
    backward_weights_kernel<<<gridDim, blockDim, 0, curr_stream>>>(kernel_in, kernel_grad_out, kernel_grad_weights, conv);

    // Check for kernel error
    ERROR_CHECK(cudaGetLastError());
}