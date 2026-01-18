#include "activations.h"
#include <stdexcept>
#include <cmath>
#include <cstdio>
#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #define CUDA_CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); \
  } while(0)
#endif

// ================= DEBUG HELPER START =================
#ifdef __CUDACC__
__global__ void debug_stats_kernel_act(const float* data, int n, float* out_stats) {
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

static void print_debug_stats_act(const char* tag, const float* d_data, int size) {
    if (size <= 0) return;
    float* d_stats;
    float h_stats[3];
    cudaMalloc(&d_stats, 3 * sizeof(float));
    debug_stats_kernel_act<<<1, 1>>>(d_data, size, d_stats);
    cudaDeviceSynchronize(); 
    cudaMemcpy(h_stats, d_stats, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[%s] Size=%d, AvgAbs=%.6f, Max=%.6f, NaNs=%d\n", tag, size, h_stats[0], h_stats[1], (int)h_stats[2]);
    cudaFree(d_stats);
}
#endif
// ================= DEBUG HELPER END =================

//  CPU 实现 
static Tensor relu_forward_cpu(const Tensor& x) {
    Tensor y(x.shape(), Device::cpu());
    const float* px = x.data();
    float* py = y.data();
    for (int64_t i = 0; i < x.numel(); ++i) py[i] = px[i] > 0.f ? px[i] : 0.f;
    return y;
}

static Tensor relu_backward_cpu(const Tensor& x, const Tensor& gy) {
    if (x.numel() != gy.numel()) throw std::runtime_error("size mismatch");
    Tensor gx(x.shape(), Device::cpu());
    const float* px = x.data();
    const float* pgy = gy.data();
    float* pgx = gx.data();
    for (int64_t i = 0; i < x.numel(); ++i) pgx[i] = (px[i] > 0.f) ? pgy[i] : 0.f;
    return gx;
}

static Tensor sigmoid_forward_cpu(const Tensor& x) {
    Tensor y(x.shape(), Device::cpu());
    const float* px = x.data();
    float* py = y.data();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float v = px[i];
        py[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return y;
}

static Tensor sigmoid_backward_cpu(const Tensor& y, const Tensor& gy) {
    if (y.numel() != gy.numel()) throw std::runtime_error("size mismatch");
    Tensor gx(y.shape(), Device::cpu());
    const float* py = y.data();
    const float* pgy = gy.data();
    float* pgx = gx.data();
    for (int64_t i = 0; i < y.numel(); ++i) {
        pgx[i] = pgy[i] * py[i] * (1.0f - py[i]); 
    }
    return gx;
}

//  CUDA 实现 
#ifdef __CUDACC__
__global__ void relu_fwd_kernel(const float* x, float* y, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

__global__ void relu_bwd_kernel(const float* x, const float* gy, float* gx, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gx[i] = (x[i] > 0.f) ? gy[i] : 0.f;
}

__global__ void sigmoid_fwd_kernel(const float* x, float* y, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 1.0f / (1.0f + __expf(-v));
    }
}

__global__ void sigmoid_bwd_kernel(const float* y, const float* gy, float* gx, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gx[i] = gy[i] * y[i] * (1.0f - y[i]);
    }
}

static int grid_for(int64_t n) {
    const int block = 256;
    return static_cast<int>((n + block - 1) / block);
}

static Tensor relu_forward_gpu(const Tensor& x) {
    Tensor y(x.shape(), x.device());
    int64_t n = x.numel();
    if (n == 0) return y;
    CUDA_CHECK(cudaSetDevice(x.device().index));
    relu_fwd_kernel<<<grid_for(n), 256>>>(x.data(), y.data(), n);
    CUDA_CHECK(cudaGetLastError());

    print_debug_stats_act("ReLU Fwd: Output", y.data(), n);
    return y;
}

static Tensor relu_backward_gpu(const Tensor& x, const Tensor& gy) {
    Tensor gx(x.shape(), x.device());
    int64_t n = x.numel();
    CUDA_CHECK(cudaSetDevice(x.device().index));

    print_debug_stats_act("ReLU Bwd: Grad Output (Incoming)", gy.data(), n);

    relu_bwd_kernel<<<grid_for(n), 256>>>(x.data(), gy.data(), gx.data(), n);
    CUDA_CHECK(cudaGetLastError());

    print_debug_stats_act("ReLU Bwd: Grad Input (Result)", gx.data(), n);
   
    return gx;
}

static Tensor sigmoid_forward_gpu(const Tensor& x) {
    Tensor y(x.shape(), x.device());
    int64_t n = x.numel();
    CUDA_CHECK(cudaSetDevice(x.device().index));
    sigmoid_fwd_kernel<<<grid_for(n), 256>>>(x.data(), y.data(), n);
    CUDA_CHECK(cudaGetLastError());
    return y;
}

static Tensor sigmoid_backward_gpu(const Tensor& y, const Tensor& gy) {
    Tensor gx(y.shape(), y.device());
    int64_t n = y.numel();
    CUDA_CHECK(cudaSetDevice(y.device().index));
    sigmoid_bwd_kernel<<<grid_for(n), 256>>>(y.data(), gy.data(), gx.data(), n);
    CUDA_CHECK(cudaGetLastError());
    return gx;
}
#endif

Tensor relu_forward(const Tensor& x) {
    if (x.device().type == DeviceType::CPU) return relu_forward_cpu(x);
#ifdef __CUDACC__
    return relu_forward_gpu(x);
#else
    throw std::runtime_error("CUDA not available");
#endif
}

Tensor relu_backward(const Tensor& x, const Tensor& grad_y) {
    if (x.device().type != grad_y.device().type) throw std::runtime_error("device mismatch");
    if (x.numel() != grad_y.numel()) throw std::runtime_error("size mismatch");
    if (x.device().type == DeviceType::CPU) return relu_backward_cpu(x, grad_y);
#ifdef __CUDACC__
    return relu_backward_gpu(x, grad_y);
#else
    throw std::runtime_error("CUDA not available");
#endif
}

Tensor sigmoid_forward(const Tensor& x) {
    if (x.device().type == DeviceType::CPU) return sigmoid_forward_cpu(x);
#ifdef __CUDACC__
    return sigmoid_forward_gpu(x);
#else
    throw std::runtime_error("CUDA not available");
#endif
}

Tensor sigmoid_backward(const Tensor& y, const Tensor& grad_y) {
    if (y.device().type != grad_y.device().type) throw std::runtime_error("device mismatch");
    if (y.numel() != grad_y.numel()) throw std::runtime_error("size mismatch");
    if (y.device().type == DeviceType::CPU) return sigmoid_backward_cpu(y, grad_y);
#ifdef __CUDACC__
    return sigmoid_backward_gpu(y, grad_y);
#else
    throw std::runtime_error("CUDA not available");
#endif
}