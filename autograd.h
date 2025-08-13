#pragma once

#include <torch/types.h>
#include <torch/autograd.h>
#include <iostream>

torch::Tensor conv2d_relu_int8_forward(
    const torch::Tensor in,
    const torch::Tensor weights,
    const torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    float x_scale, int x_zp,
    const torch::Tensor w_scale, const torch::Tensor w_zp,
    bool use_relu
);

torch::Tensor conv2d_relu_int8_input_backward(
    const torch::Tensor grad_out,
    const torch::Tensor weights,
    int stride, int padding, int dilation,
    const torch::Tensor w_scale, const torch::Tensor w_zp,
    int H_in, int W_in
);

torch::Tensor conv2d_relu_int8_weights_backward(
    const torch::Tensor in,
    const torch::Tensor grad_out,
    int stride, int padding, int dilation,
    int k_h, int k_w,
    float x_scale, int x_zp
);

class Conv2dReLUInt8Function : public torch::autograd::Function<Conv2dReLUInt8Function> {
 public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& in,
        const torch::Tensor& weights,
        const torch::Tensor& bias,
        int stride, int padding, int dilation,
        bool use_relu
    ) {
        // Quantize input and weights
        at::Tensor w_scale = torch::empty({weights.size(0)}, torch::dtype(torch::kFloat32).device(in.device()));
        at::Tensor w_zp = torch::empty({weights.size(0)}, torch::dtype(torch::kInt32).device(in.device()));
        at::Tensor w_max = weights.abs().amax({1,2,3});
        w_scale = w_max / 127.f;
        w_zp.zero_();
        auto x_max = in.abs().max().item<float>();
        float x_scale = x_max / 127.f;
        int x_zp = 0;

        at::Tensor quant_in = at::quantize_per_tensor(in, x_scale, x_zp, at::kQInt8).int_repr().contiguous();
        at::Tensor quant_weights = at::quantize_per_channel(weights, w_scale, w_zp, 0, at::kQInt8).int_repr().contiguous();

        auto out = conv2d_relu_int8_forward(quant_in, quant_weights, bias, stride, padding, dilation, x_scale, x_zp, w_scale, w_zp, use_relu);

        // Save for backward
        ctx->save_for_backward({quant_in, quant_weights, out.contiguous()});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["x_scale"] = x_scale;
        ctx->saved_data["x_zp"] = x_zp;
        ctx->saved_data["w_scale"] = w_scale.contiguous();
        ctx->saved_data["w_zp"] = w_zp.contiguous();
        ctx->saved_data["has_bias"] = bias.numel() != 0;
        ctx->saved_data["use_relu"] = use_relu;

        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outs
    ) {
        auto grad_out = grad_outs[0].contiguous();
        auto saved = ctx->get_saved_variables();
        if (ctx->saved_data["use_relu"].toBool()) {
            grad_out = (saved[2] > 0).to(grad_out.scalar_type()) * grad_out; // Apply ReLU mask
        }
        torch::Tensor grad_in = conv2d_relu_int8_input_backward(grad_out, saved[1], ctx->saved_data["stride"].toInt(), ctx->saved_data["padding"].toInt(), ctx->saved_data["dilation"].toInt(), ctx->saved_data["w_scale"].toTensor(), ctx->saved_data["w_zp"].toTensor(), saved[0].size(2), saved[0].size(3));
        torch::Tensor grad_weights = conv2d_relu_int8_weights_backward(saved[0], grad_out, ctx->saved_data["stride"].toInt(), ctx->saved_data["padding"].toInt(), ctx->saved_data["dilation"].toInt(), saved[1].size(2), saved[1].size(3), ctx->saved_data["x_scale"].toDouble(), ctx->saved_data["x_zp"].toInt());
        torch::Tensor grad_bias;
        if (ctx->saved_data["has_bias"].toBool()) {
            grad_bias = grad_out.sum({0, 2, 3});
        } else {
            grad_bias = torch::Tensor();
        }
        return {
            grad_in,
            grad_weights,
            grad_bias,
            torch::Tensor(), // stride
            torch::Tensor(), // padding
            torch::Tensor(), // dilation
            torch::Tensor()  // use_relu
        };
    }
};