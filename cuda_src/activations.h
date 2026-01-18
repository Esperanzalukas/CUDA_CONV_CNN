#pragma once
#include "tensor.h"

// ReLU
Tensor relu_forward(const Tensor& x);
Tensor relu_backward(const Tensor& x, const Tensor& grad_y); 

// Sigmoid
Tensor sigmoid_forward(const Tensor& x);
Tensor sigmoid_backward(const Tensor& y, const Tensor& grad_y); 