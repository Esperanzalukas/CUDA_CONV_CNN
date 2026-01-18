#include "tensor.h"
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); \
} while(0)

static int64_t product(const std::vector<int64_t>& v) {
    if (v.empty()) return 0;
    int64_t p = 1;
    for (auto x : v) p *= x;
    return p;
}

Tensor::Tensor(const std::vector<int64_t>& shape, Device device)
: shape_(shape), numel_(product(shape)), device_(device) {
    allocate();
}

Tensor::~Tensor() { deallocate(); }

Tensor::Tensor(Tensor&& other) noexcept {
    shape_   = std::move(other.shape_);
    numel_   = other.numel_;
    device_  = other.device_;
    storage_ = other.storage_;
    other.numel_ = 0;
    other.storage_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_   = std::move(other.shape_);
        numel_   = other.numel_;
        device_  = other.device_;
        storage_ = other.storage_;
        other.numel_ = 0;
        other.storage_ = nullptr;
    }
    return *this;
}

void Tensor::allocate() {
    if (numel_ <= 0) return;
    size_t nbytes = static_cast<size_t>(numel_) * sizeof(float);
    if (device_.type == DeviceType::CPU) {
        storage_ = std::malloc(nbytes);
        if (!storage_) throw std::bad_alloc();
    } else {
        CUDA_CHECK(cudaSetDevice(device_.index));
        CUDA_CHECK(cudaMalloc(&storage_, nbytes));
    }
}

void Tensor::deallocate() {
    if (!storage_) return;
    if (device_.type == DeviceType::CPU) {
        std::free(storage_);
    } else {
        CUDA_CHECK(cudaSetDevice(device_.index));
        CUDA_CHECK(cudaFree(storage_));
    }
    storage_ = nullptr;
}

float* Tensor::data() { return reinterpret_cast<float*>(storage_); }
const float* Tensor::data() const { return reinterpret_cast<const float*>(storage_); }

void Tensor::copy_to(Tensor& dst) const {
    if (numel_ != dst.numel_) throw std::runtime_error("size mismatch in copy_to");
    size_t nbytes = static_cast<size_t>(numel_) * sizeof(float);

    if (device_.type == DeviceType::CPU && dst.device_.type == DeviceType::CPU) {
        std::memcpy(dst.storage_, storage_, nbytes);
    } else if (device_.type == DeviceType::CPU && dst.device_.type == DeviceType::GPU) {
        CUDA_CHECK(cudaSetDevice(dst.device_.index));
        CUDA_CHECK(cudaMemcpy(dst.storage_, storage_, nbytes, cudaMemcpyHostToDevice));
    } else if (device_.type == DeviceType::GPU && dst.device_.type == DeviceType::CPU) {
        CUDA_CHECK(cudaSetDevice(device_.index));
        CUDA_CHECK(cudaMemcpy(dst.storage_, storage_, nbytes, cudaMemcpyDeviceToHost));
    } else if (device_.type == DeviceType::GPU && dst.device_.type == DeviceType::GPU) {
        CUDA_CHECK(cudaSetDevice(device_.index));
        CUDA_CHECK(cudaMemcpy(dst.storage_, storage_, nbytes, cudaMemcpyDeviceToDevice));
    }
}

Tensor Tensor::cpu() const {
    Tensor out(shape_, Device::cpu());
    if (numel_ > 0) copy_to(out);
    return out;
}

Tensor Tensor::gpu(int idx) const {
    Tensor out(shape_, Device::gpu(idx));
    if (numel_ > 0) copy_to(out);
    return out;
}

void Tensor::fill(float v) {
    if (numel_ <= 0) return;
    if (device_.type == DeviceType::CPU) {
        float* p = data();
        for (int64_t i = 0; i < numel_; ++i) p[i] = v;
    } else {
        std::vector<float> tmp(static_cast<size_t>(numel_), v);
        Tensor host(shape_, Device::cpu());
        std::memcpy(host.data(), tmp.data(), tmp.size() * sizeof(float));
        host.copy_to(*this);
    }
}