#include <stdint.h>

struct conv2d {
    int C_in, C_out;
    int H_in, H_out;
    int W_in, W_out;
    int k_h, k_w;
    int B, stride, padding, dilation, groups;

    int8_t* weights;
    float* bias;

    float x_scale, w_scale, y_scale;
    int x_zp, w_zp, y_zp;
};