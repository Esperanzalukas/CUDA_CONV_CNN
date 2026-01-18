#include "conv_layer.h"
#include <iostream>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// =========================================================================
// Debug Helpers (Unique name to avoid linker errors)
// =========================================================================

__global__ void debug_stats_kernel_conv(const float* data, int n, float* out_stats) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        float max_val = -1e9f;
        int nans = 0;
        for(int i=0; i<n; i++) {
            float v = data[i];
            if (isnan(v)) nans++;
            else {
                sum += (v > 0 ? v : -v); // abs sum
                if(v > max_val) max_val = v;
            }
        }
        out_stats[0] = sum / (n > 0 ? n : 1);
        out_stats[1] = max_val;
        out_stats[2] = (float)nans;
    }
}

static void print_debug_stats_conv(const char* tag, const float* d_data, int size) {
    if (size <= 0) return; 
    float* d_stats;
    float h_stats[3];
    cudaMalloc(&d_stats, 3 * sizeof(float));
    debug_stats_kernel_conv<<<1, 1>>>(d_data, size, d_stats);
    cudaDeviceSynchronize();
    cudaMemcpy(h_stats, d_stats, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[%s] Size=%d, AvgAbs=%.6f, Max=%.6f, NaNs=%d\n", tag, size, h_stats[0], h_stats[1], (int)h_stats[2]);
    cudaFree(d_stats);
}

// =========================================================================
// IM2COL & COL2IM Implementation
// =========================================================================

__global__ void im2col_kernel(const int n, const float* data_im, 
                              const int height, const int width, const int ksize, 
                              const int pad, const int stride, 
                              const int height_col, const int width_col, 
                              float* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    int m = height_col * width_col; 
    
    int spatial_idx = index % m;
    int kernel_idx = index / m;
    
    int w_out = spatial_idx % width_col;
    int h_out = spatial_idx / width_col;
    
    int k_sq = ksize * ksize;
    int channel_in = kernel_idx / k_sq;
    int k_rem = kernel_idx % k_sq;
    int u = k_rem / ksize;
    int v = k_rem % ksize;
    
    int h_in = h_out * stride - pad + u;
    int w_in = w_out * stride - pad + v;
    
    float val = 0.0f;
    if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
        val = data_im[(channel_in * height + h_in) * width + w_in];
    }
    
    data_col[index] = val;
}

void im2col_gpu(float* d_input, float* d_col_buffer,
                int in_channels, int height, int width, int ksize, int padding, 
                int stride, int height_out, int width_out) {
    int m = height_out * width_out;
    int k = in_channels * ksize * ksize;
    int num_kernels = m * k;
    
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;
    
    im2col_kernel<<<blocks, threads>>>(num_kernels, d_input, height, width, ksize, padding, 
                                       stride, height_out, width_out, d_col_buffer);
}

__global__ void col2im_kernel(const int n, const float* data_col,
                              const int height, const int width, const int ksize, 
                              const int pad, const int stride, 
                              const int height_col, const int width_col,
                              float* data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    
    int m = height_col * width_col;
    
    int spatial_idx = index % m;
    int kernel_idx = index / m;
    
    int w_out = spatial_idx % width_col;
    int h_out = spatial_idx / width_col;
    
    int k_sq = ksize * ksize;
    int channel_in = kernel_idx / k_sq;
    int k_rem = kernel_idx % k_sq;
    int u = k_rem / ksize;
    int v = k_rem % ksize;
    
    int h_in = h_out * stride - pad + u;
    int w_in = w_out * stride - pad + v;
    
    if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
        float val = data_col[index];
        atomicAdd(&data_im[(channel_in * height + h_in) * width + w_in], val);
    }
}

void col2im_gpu(float* d_col_buffer, float* d_grad_input,
                int in_channels, int height, int width, int ksize, int padding, 
                int stride, int height_out, int width_out) {
    int m = height_out * width_out;
    int k = in_channels * ksize * ksize;
    int num_kernels = m * k;
    
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;
    
    col2im_kernel<<<blocks, threads>>>(num_kernels, d_col_buffer, height, width, ksize, padding, 
                                       stride, height_out, width_out, d_grad_input);
}


// =========================================================================
// Forward & Backward
// =========================================================================

void forward_conv(float* d_input, float* d_output, float* d_weight, float* d_col_buffer,
                  int batch_size, int in_channels, int out_channels, 
                  int height, int width, int ksize, int stride, int padding,
                  cublasHandle_t handle) {

    int height_out = (height + 2 * padding - ksize) / stride + 1;
    int width_out = (width + 2 * padding - ksize) / stride + 1;

    int m = height_out * width_out; 
    int n = out_channels;           
    int k = in_channels * ksize * ksize; 

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        im2col_gpu(d_input + b * in_channels * height * width, 
                   d_col_buffer, 
                   in_channels, height, width, ksize, padding, stride, height_out, width_out);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    d_col_buffer, m,  
                    d_weight, k,      
                    &beta,
                    d_output + b * out_channels * height_out * width_out, m);
    }

    //print_debug_stats_conv("Conv Fwd Input", d_input, batch_size * in_channels * height * width);
    //print_debug_stats_conv("Conv Fwd Weight", d_weight, out_channels * in_channels * ksize * ksize);
    //print_debug_stats_conv("Conv Fwd Output", d_output, batch_size * out_channels * height_out * width_out);

}

void backward_conv(float* d_input, float* d_grad_output, float* d_weight, 
                   float* d_col_buffer, float* d_grad_input, float* d_grad_weight,
                   int batch_size, int in_channels, int out_channels, 
                   int height, int width, int ksize, int stride, int padding,
                   cublasHandle_t handle) {

    int height_out = (height + 2 * padding - ksize) / stride + 1;
    int width_out = (width + 2 * padding - ksize) / stride + 1;
    
    int m = height_out * width_out;
    int n = out_channels;
    int k = in_channels * ksize * ksize;

    float alpha = 1.0f;
    float beta = 0.0f;
    float beta_one = 1.0f; 

    //print_debug_stats_conv("Conv Bwd: Input Grad_Output", d_grad_output, batch_size * n * m);

    CHECK_CUDA(cudaMemset(d_grad_weight, 0, sizeof(float) * out_channels * k));
    CHECK_CUDA(cudaMemset(d_grad_input, 0, sizeof(float) * batch_size * in_channels * height * width));
    
    for(int i=0; i<batch_size; i++) {
        im2col_gpu(d_input + i * in_channels * height * width, d_col_buffer,
                   in_channels, height, width, ksize, padding, stride, height_out, width_out);
        
        // dW
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    k, n, m,
                    &alpha,
                    d_col_buffer, m,
                    d_grad_output + i * n * m, m,
                    &beta_one, 
                    d_grad_weight, k);

        // dX_col
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, k, n,
                    &alpha,
                    d_grad_output + i * n * m, m, 
                    d_weight, k, 
                    &beta, 
                    d_col_buffer, m);
        
        col2im_gpu(d_col_buffer, 
                   d_grad_input + i * in_channels * height * width,
                   in_channels, height, width, ksize, padding, stride, height_out, width_out);
    }

    //print_debug_stats_conv("Conv Bwd: Result dInput", d_grad_input, batch_size * in_channels * height * width);
    //print_debug_stats_conv("Conv Bwd: Result dWeight", d_grad_weight, out_channels * k);
}