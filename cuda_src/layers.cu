#include "layers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cstdio>

// ================= DEBUG HELPER START =================
__global__ void debug_stats_kernel_layers(const float* data, int n, float* out_stats) {
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

static void print_debug_stats_layers(const char* tag, const float* d_data, int size) {
    if (size <= 0) return;
    float* d_stats;
    float h_stats[3];
    cudaMalloc(&d_stats, 3 * sizeof(float));
    debug_stats_kernel_layers<<<1, 1>>>(d_data, size, d_stats);
    cudaDeviceSynchronize(); 
    cudaMemcpy(h_stats, d_stats, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[%s] Size=%d, AvgAbs=%.6f, Max=%.6f, NaNs=%d\n", tag, size, h_stats[0], h_stats[1], (int)h_stats[2]);
    cudaFree(d_stats);
}
// ================= DEBUG HELPER END =================

inline void get_grid(int n, int& blocks, int& threads) {
    threads = 256;
    blocks = (n + threads - 1) / threads;
}

// ==========================================
// 1. Sigmoid
// ==========================================
__global__ void sigmoid_fwd_kernel(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

void sigmoid_forward(const Tensor& input, Tensor& output) {
    int n = input.numel();
    int b, t; get_grid(n, b, t);
    sigmoid_fwd_kernel<<<b, t>>>(n, input.data(), output.data());
}

__global__ void sigmoid_bwd_kernel(int n, const float* y, const float* grad_y, float* grad_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y_val = y[i];
        grad_x[i] = grad_y[i] * y_val * (1.0f - y_val);
    }
}

void sigmoid_backward(const Tensor& output, const Tensor& grad_output, Tensor& grad_input) {
    int n = output.numel();
    int b, t; get_grid(n, b, t);
    sigmoid_bwd_kernel<<<b, t>>>(n, output.data(), grad_output.data(), grad_input.data());
}

// ==========================================
// 2. ReLU
// ==========================================
__global__ void relu_fwd_kernel(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void relu_forward(const Tensor& input, Tensor& output) {
    int n = input.numel();
    int b, t; get_grid(n, b, t);
    relu_fwd_kernel<<<b, t>>>(n, input.data(), output.data());
}

__global__ void relu_bwd_kernel(int n, const float* x, const float* grad_y, float* grad_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_x[i] = x[i] > 0.0f ? grad_y[i] : 0.0f;
}

void relu_backward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
    int n = input.numel();
    int b, t; get_grid(n, b, t);
    relu_bwd_kernel<<<b, t>>>(n, input.data(), grad_output.data(), grad_input.data());
}

// ==========================================
// 3. Linear (Fully Connected)
// ==========================================
__global__ void add_bias_kernel(const float* bias, float* output, int N, int OutFeatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * OutFeatures) {
        int col = idx % OutFeatures;
        output[idx] += bias[col];
    }
}

void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    int N = input.shape()[0];
    int InF = input.shape()[1];
    int OutF = weight.shape()[0];

    //print_debug_stats_layers("Linear Fwd: Input", input.data(), input.numel());
    //print_debug_stats_layers("Linear Fwd: Weight", weight.data(), weight.numel());

    cublasHandle_t handle; cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                OutF, N, InF,
                &alpha,
                weight.data(), InF,  
                input.data(), InF,   
                &beta,
                output.data(), OutF);

    int b, t; get_grid(N * OutF, b, t);
    add_bias_kernel<<<b, t>>>(bias.data(), output.data(), N, OutF);
    
    //print_debug_stats_layers("Linear Fwd: Output", output.data(), output.numel());
    cublasDestroy(handle);
}

__global__ void reduce_sum_bias_kernel(const float* grad_out, float* grad_bias, int N, int OutF) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < OutF) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) sum += grad_out[i * OutF + col];
        grad_bias[col] = sum;
    }
}

void linear_backward(const Tensor& input, const Tensor& weight, const Tensor& grad_output, 
                     Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias) {
    int N = input.shape()[0];
    int InF = input.shape()[1];
    int OutF = weight.shape()[0];
    
    cublasHandle_t handle; cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    //print_debug_stats_layers("Linear Bwd: Grad Output", grad_output.data(), grad_output.numel());

    // 1. dInput = dOutput * W
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                InF, N, OutF,
                &alpha,
                weight.data(), InF,
                grad_output.data(), OutF,
                &beta,
                grad_input.data(), InF);

    // 2. dWeight = dOutput^T * Input
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                InF, OutF, N,
                &alpha,
                input.data(), InF,
                grad_output.data(), OutF,
                &beta,
                grad_weight.data(), InF);

    // 3. dBias
    int b, t; get_grid(OutF, b, t);
    reduce_sum_bias_kernel<<<b, t>>>(grad_output.data(), grad_bias.data(), N, OutF);

    //print_debug_stats_layers("Linear Bwd: Grad Input", grad_input.data(), grad_input.numel());
    //print_debug_stats_layers("Linear Bwd: Grad Weight", grad_weight.data(), grad_weight.numel());
    
    cublasDestroy(handle);
}

// ==========================================
// 4. Convolution (Only wrappers here if not using conv_layer.cu directly)
// ... keeping other functions as is ... 
// (由于 conv2d_forward/backward 主要在 conv_layer.cu 实现了，这里可能是遗留代码或重复定义)
// (如果你之前的 layers.cu 里还有 conv 代码，保留原样即可，如果它是调用 conv_layer 的接口)
// 根据你提供的 layers.cu，它里面确实有 conv2d_forward 的实现，这可能是重复的根源之一。
// **注意**：如果 conv_layer.cu 和 layers.cu 都有 conv2d_forward，链接也会报错。
// 我这里保留你 layers.cu 的内容不变，只改 debug 函数名。
// ==========================================

__global__ void im2col_kernel_layers(const int n, const float* data_im,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col * width_col + h_out * width_col + w_out);
    const float* data_im_ptr = data_im + (channel_in * height * width);

    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            int h = h_in + i;
            int w = w_in + j;
            *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im_ptr[h * width + w] : 0;
            data_col_ptr += height_col * width_col;
        }
    }
}

__global__ void col2im_kernel_layers(const int n, const float* data_col,
                              const int height, const int width, const int channels,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int height_col, const int width_col,
                              float* data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    float val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);

    int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);

    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            int h_k = h - h_col * stride_h;
            int w_k = w - w_col * stride_w;
            if (h_k % 1 == 0 && w_k % 1 == 0) { 
                int data_col_index = (((c * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) * width_col + w_col;
                val += data_col[data_col_index];
            }
        }
    }
    data_im[index] = val;
}

void conv2d_forward(const Tensor& input, const Tensor& weight, Tensor& output, int stride, int padding) {
    int batch = input.shape()[0];
    int in_c = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];
    int out_c = weight.shape()[0];
    int k_h = weight.shape()[2];
    int k_w = weight.shape()[3];
    int out_h = output.shape()[2];
    int out_w = output.shape()[3];

    cublasHandle_t handle; cublasCreate(&handle);
    float* d_col;
    cudaMalloc(&d_col, in_c * k_h * k_w * out_h * out_w * sizeof(float));

    for (int n = 0; n < batch; ++n) {
        int num_kernels = in_c * out_h * out_w;
        int b, t; get_grid(num_kernels, b, t);
        im2col_kernel_layers<<<b, t>>>(num_kernels, input.data() + n*in_c*in_h*in_w, 
                                in_h, in_w, in_c, k_h, k_w, padding, padding, stride, stride,
                                out_h, out_w, d_col);
        
        float alpha = 1.0f, beta = 0.0f;
        int m = out_h * out_w;
        int k = in_c * k_h * k_w;
        int n_dim = out_c;
        
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n_dim, k,
                    &alpha, 
                    d_col, m,
                    weight.data(), k, 
                    &beta, 
                    output.data() + n*out_c*out_h*out_w, m);
    }
    cudaFree(d_col);
    cublasDestroy(handle);
}

void conv2d_backward(const Tensor& input, const Tensor& weight, const Tensor& grad_output, 
                     Tensor& grad_input, Tensor& grad_weight, int stride, int padding) {
    int batch = input.shape()[0];
    int in_c = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];
    int out_c = weight.shape()[0];
    int k_h = weight.shape()[2];
    int k_w = weight.shape()[3];
    int out_h = grad_output.shape()[2];
    int out_w = grad_output.shape()[3];

    cublasHandle_t handle; cublasCreate(&handle);
    float* d_col;
    int col_size = in_c * k_h * k_w * out_h * out_w;
    cudaMalloc(&d_col, col_size * sizeof(float));

    int m = out_h * out_w;
    int k = in_c * k_h * k_w;
    int n_dim = out_c; 

    float alpha = 1.0f, beta = 0.0f, beta_one = 1.0f;

    grad_weight.fill(0.0f); 

    for (int n = 0; n < batch; ++n) {
        int num_kernels = in_c * out_h * out_w;
        int b, t; get_grid(num_kernels, b, t);
        im2col_kernel_layers<<<b, t>>>(num_kernels, input.data() + n*in_c*in_h*in_w, 
                                in_h, in_w, in_c, k_h, k_w, padding, padding, stride, stride,
                                out_h, out_w, d_col);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    k, n_dim, m,
                    &alpha,
                    d_col, m,
                    grad_output.data() + n*out_c*m, m,
                    &beta_one, 
                    grad_weight.data(), k);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, k, n_dim,
                    &alpha,
                    grad_output.data() + n*out_c*m, m,
                    weight.data(), k,
                    &beta,
                    d_col, m);
        
        int num_im = in_c * in_h * in_w; 
        get_grid(num_im, b, t);
        col2im_kernel_layers<<<b, t>>>(num_im, d_col, in_h, in_w, in_c, k_h, k_w, 
                                padding, padding, stride, stride, out_h, out_w, 
                                grad_input.data() + n*in_c*in_h*in_w);
    }
    cudaFree(d_col);
    cublasDestroy(handle);
}

// ... Max Pooling, Softmax etc ...
// (保持剩余部分不变)

__global__ void maxpool_fwd_kernel(int nthreads, const float* bottom, int c, int h, int w, 
                                   int ph, int pw, int kh, int kw, int sh, int sw, int pad,
                                   float* top, float* mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nthreads) return;
    int pw_idx = idx % pw;
    int ph_idx = (idx / pw) % ph;
    int c_idx = (idx / pw / ph) % c;
    int n_idx = idx / pw / ph / c;

    int hstart = ph_idx * sh - pad;
    int wstart = pw_idx * sw - pad;
    int hend = min(hstart + kh, h);
    int wend = min(wstart + kw, w);
    hstart = max(hstart, 0); wstart = max(wstart, 0);

    float maxval = -FLT_MAX;
    int maxidx = -1;
    const float* src = bottom + (n_idx*c + c_idx)*h*w;
    for (int i = hstart; i < hend; ++i) {
        for (int j = wstart; j < wend; ++j) {
            if (src[i*w + j] > maxval) {
                maxval = src[i*w + j];
                maxidx = i*w + j;
            }
        }
    }
    top[idx] = maxval;
    mask[idx] = (float)(maxidx + (n_idx*c + c_idx)*h*w);
}

void maxpool_forward(const Tensor& input, Tensor& output, Tensor& mask, int k, int s, int p) {
    int n = output.numel();
    int b, t; get_grid(n, b, t);
    maxpool_fwd_kernel<<<b, t>>>(n, input.data(), input.shape()[1], input.shape()[2], input.shape()[3],
                                 output.shape()[2], output.shape()[3], k, k, s, s, p, 
                                 output.data(), mask.data());
}

__global__ void maxpool_bwd_kernel(int n, const float* top_diff, const float* mask, float* bottom_diff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(bottom_diff + (int)mask[i], top_diff[i]);
}

void maxpool_backward(const Tensor& grad_output, const Tensor& mask, Tensor& grad_input, int k, int s, int p) {
    grad_input.fill(0.0f);
    int n = grad_output.numel();
    int b, t; get_grid(n, b, t);
    maxpool_bwd_kernel<<<b, t>>>(n, grad_output.data(), mask.data(), grad_input.data());
}

__global__ void softmax_kernel(const float* x, float* y, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* row_x = x + i*C;
        float* row_y = y + i*C;
        float maxv = -FLT_MAX;
        for(int j=0; j<C; ++j) maxv = max(maxv, row_x[j]);
        float sum = 0.0f;
        for(int j=0; j<C; ++j) {
            row_y[j] = expf(row_x[j] - maxv);
            sum += row_y[j];
        }
        for(int j=0; j<C; ++j) row_y[j] /= sum;
    }
}

void softmax_forward(const Tensor& input, Tensor& output) {
    int N = input.shape()[0];
    int C = input.shape()[1];
    int b, t; get_grid(N, b, t);
    softmax_kernel<<<b, t>>>(input.data(), output.data(), N, C);
}

__global__ void ce_kernel(const float* p, const float* label, float* loss, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int l = (int)label[i];
        atomicAdd(loss, -logf(p[i*C + l] + 1e-7f));
    }
}

float cross_entropy_loss(const Tensor& probs, const Tensor& labels) {
    int N = probs.shape()[0];
    int C = probs.shape()[1];
    float h_loss = 0.0f;
    float* d_loss; cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);
    
    int b, t; get_grid(N, b, t);
    ce_kernel<<<b, t>>>(probs.data(), labels.data(), d_loss, N, C);
    
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    return h_loss / N;
}

__global__ void ce_bwd_kernel(const float* p, const float* label, float* grad, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N*C) {
        int n = idx / C;
        int c = idx % C;
        int l = (int)label[n];
        grad[idx] = (p[idx] - (c == l ? 1.0f : 0.0f)) / N;
    }
}

void cross_entropy_backward(const Tensor& probs, const Tensor& labels, Tensor& grad_input) {
    int N = probs.shape()[0];
    int C = probs.shape()[1];
    int b, t; get_grid(N*C, b, t);
    ce_bwd_kernel<<<b, t>>>(probs.data(), labels.data(), grad_input.data(), N, C);
}