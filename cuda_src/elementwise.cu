#include "elementwise.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); \
} while(0)

__global__ void add_kernel(const float* a, const float* b, float* c, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] = grad_out[i];
        grad_b[i] = grad_out[i];
    }
}

void elementwise_add(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.numel() != b.numel() || a.numel() != c.numel()) {
        throw std::runtime_error("Tensor size mismatch in elementwise_add");
    }
    
    if (a.device().type != DeviceType::GPU || 
        b.device().type != DeviceType::GPU || 
        c.device().type != DeviceType::GPU) {
        throw std::runtime_error("elementwise_add requires GPU tensors");
    }
    
    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    CUDA_CHECK(cudaSetDevice(a.device().index));
    add_kernel<<<blocks, threads>>>(a.data(), b.data(), c.data(), n);
    CUDA_CHECK(cudaGetLastError());
}

void elementwise_add_backward(const Tensor& grad_out, Tensor& grad_a, Tensor& grad_b) {
    if (grad_out.device().type != DeviceType::GPU) {
        throw std::runtime_error("elementwise_add_backward requires GPU tensor");
    }
    
    int64_t n = grad_out.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    CUDA_CHECK(cudaSetDevice(grad_out.device().index));
    add_backward_kernel<<<blocks, threads>>>(grad_out.data(), grad_a.data(), grad_b.data(), n);
    CUDA_CHECK(cudaGetLastError());
}