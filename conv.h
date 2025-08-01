#include <stdint.h>

struct conv2d {
    int C_in, C_out;
    int H_in, H_out;
    int W_in, W_out;
    int k_h, k_w;
    int batch_size, stride, padding, dilation;

    int8_t* weights;
    float* bias;

    float x_scale, w_scale, y_scale;
    int x_zp, w_zp, y_zp;
};

__global__ void kernel(const int8_t* __restrict__ in, int8_t* __restrict__ out, conv2d& conv);