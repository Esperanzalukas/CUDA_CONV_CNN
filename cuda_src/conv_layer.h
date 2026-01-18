#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// 前向传播
// input: [batch, in_c, h, w]
// output: [batch, out_c, h_out, w_out]
void forward_conv(float* d_input, float* d_output, float* d_weight, float* d_col_buffer,
                  int batch_size, int in_channels, int out_channels, 
                  int height, int width, int ksize, int stride, int padding,
                  cublasHandle_t handle);

// 反向传播
void backward_conv(float* d_input, float* d_grad_output, float* d_weight, 
                   float* d_col_buffer, float* d_grad_input, float* d_grad_weight,
                   int batch_size, int in_channels, int out_channels, 
                   int height, int width, int ksize, int stride, int padding,
                   cublasHandle_t handle);

#endif