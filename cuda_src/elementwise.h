#pragma once
#include "tensor.h"

void elementwise_add(const Tensor& a, const Tensor& b, Tensor& c);
void elementwise_add_backward(const Tensor& grad_out, Tensor& grad_a, Tensor& grad_b);