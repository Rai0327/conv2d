#include <torch/types.h>
#include <torch/autograd.h>

torch::Tensor conv2d_relu_int8_forward(
    const torch::Tensor in,
    const torch::Tensor weights,
    const torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);

torch::Tensor conv2d_relu_int8_input_backward(
    const torch::Tensor grad_out,
    const torch::Tensor weights,
    int stride, int padding, int dilation
);

torch::Tensor conv2d_relu_int8_weights_backward(
    const torch::Tensor in,
    const torch::Tensor grad_out,
    int stride, int padding, int dilation,
    int k_h, int k_w
);

class Conv2dReLUInt8Function : public torch::autograd::Function<Conv2dReLUInt8Function> {
 public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& in,
        const torch::Tensor& weights,
        const torch::Tensor& bias,
        int stride, int padding, int dilation
    ) {
        ctx->save_for_backward({in, weights});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;

        return conv2d_relu_int8_forward(in, weights, bias, stride, padding, dilation);
    }

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& grad_out
    ) {
        auto saved = ctx->get_saved_variables();

        torch::Tensor grad_in = conv2d_relu_int8_input_backward(grad_out, saved[1], ctx->saved_data["stride"].toInt(), ctx->saved_data["padding"].toInt(), ctx->saved_data["dilation"].toInt());
        torch::Tensor grad_weights = conv2d_relu_int8_weights_backward(saved[0], grad_out, ctx->saved_data["stride"].toInt(), ctx->saved_data["padding"].toInt(), ctx->saved_data["dilation"].toInt(), saved[1].size(2), saved[1].size(3));
        torch::Tensor grad_bias = grad_out.sum({0, 2, 3});

        return {
            grad_in,
            grad_weights,
            grad_bias,
            torch::Tensor(), // stride
            torch::Tensor(), // padding
            torch::Tensor()  // dilation
        };
    }
};