#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>

enum class DeviceType { CPU, GPU };

struct Device {
    DeviceType type;
    int index;

    Device() : type(DeviceType::CPU), index(0) {} 

    Device(DeviceType t, int i = 0) : type(t), index(i) {}
    static Device cpu() { return Device(DeviceType::CPU); }
    static Device gpu(int i = 0) { return Device(DeviceType::GPU, i); }
};

class Tensor {
public:
    Tensor(const std::vector<int64_t>& shape, Device device);
    ~Tensor();
    
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void fill(float v);
    Tensor cpu() const;
    Tensor gpu(int idx = 0) const;
    
    float* data();
    const float* data() const;
    
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t numel() const { return numel_; }
    const Device& device() const { return device_; }

private:
    void allocate();
    void deallocate();
    void copy_to(Tensor& dst) const;

    std::vector<int64_t> shape_;
    int64_t numel_;
    Device device_;
    void* storage_ = nullptr;
};