#pragma once
#include "tensor.h"

// 1. Activation: Sigmoid
void sigmoid_forward(const Tensor& input, Tensor& output);
// Sigmoid Backward: grad_input = grad_output * y * (1-y)
void sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input);

// 2. Activation: ReLU
void relu_forward(const Tensor& input, Tensor& output);
void relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);

// 3. Linear (Fully Connected)
// Y = X * W^T + b
void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);
// Linear Backward involves computing dX, dW, db
void linear_backward(const Tensor& input, const Tensor& weight, const Tensor& grad_output, 
                     Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias);

// 4. Convolution
void conv2d_forward(const Tensor& input, const Tensor& weight, Tensor& output, int stride, int padding);

// 5. Max Pooling
void maxpool_forward(const Tensor& input, Tensor& output, Tensor& mask, 
                     int k, int s, int p);
void maxpool_backward(const Tensor& grad_output, const Tensor& mask, Tensor& grad_input, 
                      int k, int s, int p);

// 6. Softmax
void softmax_forward(const Tensor& input, Tensor& output);

// 7. Cross Entropy Loss
float cross_entropy_loss(const Tensor& probs, const Tensor& labels);
void cross_entropy_backward(const Tensor& probs, const Tensor& labels, Tensor& grad_input);