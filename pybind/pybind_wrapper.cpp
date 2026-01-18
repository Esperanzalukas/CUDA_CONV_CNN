#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   
#include <pybind11/numpy.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <vector>
#include <utility>          
#include <tuple>            
#include <stdexcept>
#include <iostream>
#include <memory> 
#include <string>

#include "tensor.h"
#include "layers.h"
#include "conv_layer.h"  
#include "elementwise.h"
#include "activations.h"

namespace py = pybind11;
using namespace std;

// === 错误检查辅助函数 ===
void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::string s = "CUDA Error in " + std::string(msg) + ": " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(s);
    }
}

// === 全局 Handle ===
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    return handle;
}

// === 替换原有的 from_numpy 函数 ===
std::shared_ptr<Tensor> from_numpy(py::array_t<float> array, Device device) {
    py::buffer_info buf = array.request();
    if (buf.ndim == 0) throw std::runtime_error("Input array is empty");
    
    std::vector<int64_t> shape;
    for (auto s : buf.shape) shape.push_back(static_cast<int64_t>(s));
    
    auto t = std::make_shared<Tensor>(shape, device);
    size_t nbytes = buf.size * sizeof(float);
    
    // Debug: 检查数据有效性
    float* host_ptr = static_cast<float*>(buf.ptr);
    
    // 检查总数，防止越界
    size_t check_count = std::min((size_t)10, (size_t)buf.size);
    float sum_abs = 0.0f;
    for(size_t i=0; i < check_count; i++) {
        sum_abs += std::abs(host_ptr[i]);
    }
    
    // 如果发现异常，打印详细信息
    if (sum_abs > 1000.0f || std::isnan(sum_abs)) {
        printf("\n=== C++ CRITICAL WARNING ===\n");
        printf("from_numpy received garbage!\n");
        printf("  Shape: [");
        for(auto s : buf.shape) printf("%ld ", s);
        printf("]\n");
        printf("  Itemsize: %ld (Expect 4)\n", buf.itemsize);
        printf("  Pointer: %p\n", buf.ptr);
        printf("  First 10 values: ");
        for(size_t i=0; i<check_count; i++) printf("%e ", host_ptr[i]);
        printf("\n============================\n");
    }

    if (device.type == DeviceType::CPU) {
        std::memcpy(t->data(), host_ptr, nbytes);
    } else {
        check_cuda(cudaSetDevice(device.index), "from_numpy set device");
        check_cuda(cudaMemcpy(t->data(), host_ptr, nbytes, cudaMemcpyHostToDevice), "from_numpy memcpy");
        check_cuda(cudaDeviceSynchronize(), "from_numpy sync");
    }
    return t;
}

py::array_t<float> to_numpy(const Tensor& t) {
    std::vector<int64_t> shape = t.shape();
    std::vector<ssize_t> strides;
    ssize_t stride = sizeof(float);
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= shape[i];
    }
    py::array_t<float> result(shape, strides);
    py::buffer_info buf = result.request();
    
    if (t.device().type == DeviceType::CPU) {
         std::memcpy(buf.ptr, t.data(), t.numel() * sizeof(float));
    } else {
         check_cuda(cudaSetDevice(t.device().index), "to_numpy set device");
         check_cuda(cudaMemcpy(buf.ptr, t.data(), t.numel() * sizeof(float), cudaMemcpyDeviceToHost), "to_numpy memcpy");
         check_cuda(cudaDeviceSynchronize(), "to_numpy sync");
    }
    return result;
}

// 深度拷贝 helper
void safe_copy(const Tensor& src, Tensor& dst) {
    size_t nbytes = src.numel() * sizeof(float);
    if (src.device().type == DeviceType::GPU && dst.device().type == DeviceType::GPU) {
        check_cuda(cudaSetDevice(src.device().index), "safe_copy set device");
        check_cuda(cudaMemcpy(dst.data(), src.data(), nbytes, cudaMemcpyDeviceToDevice), "safe_copy p2p");
    } else {
        cudaMemcpy(dst.data(), src.data(), nbytes, cudaMemcpyDefault);
    }
}

PYBIND11_MODULE(my_deep_lib, m) {
    m.doc() = "Deep Learning Library";

    // 1. Device
    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(), py::arg("type"), py::arg("index")=0)
        .def_static("cpu", &Device::cpu)
        .def_static("gpu", &Device::gpu)
        .def_readwrite("type", &Device::type)
        .def_readwrite("id", &Device::index);

    // 2. DeviceType Enum
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("GPU", DeviceType::GPU)
        .export_values();

    // 3. Tensor
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, Device>())
        .def_static("from_numpy", &from_numpy)
        .def("to_numpy", &to_numpy)
        .def("fill", &Tensor::fill)
        .def_property_readonly("shape", &Tensor::shape)
        .def("numel", &Tensor::numel)
        .def("device", &Tensor::device)
        .def("reshape", [](Tensor& t, std::vector<int64_t> new_shape) {
            int64_t total = t.numel();
            int64_t infer_idx = -1;
            int64_t new_count = 1;
            for(size_t i=0; i<new_shape.size(); ++i) {
                if(new_shape[i] == -1) infer_idx=i;
                else new_count *= new_shape[i];
            }
            if(infer_idx != -1) new_shape[infer_idx] = total/new_count;
            
            auto out = std::make_shared<Tensor>(new_shape, t.device());
            safe_copy(t, *out);
            return out; 
        });

    // 4. Layers
    
    // --- Conv2D ---
    m.def("conv2d_forward", [](const Tensor& input, const Tensor& weight, int stride, int padding) {
        int batch = input.shape()[0];
        int in_c = input.shape()[1];
        int h = input.shape()[2];
        int w = input.shape()[3];
        int out_c = weight.shape()[0];
        int ksize = weight.shape()[2];
        
        int h_out = (h + 2 * padding - ksize) / stride + 1;
        int w_out = (w + 2 * padding - ksize) / stride + 1;
        
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_c, h_out, w_out}, input.device());
        
        // 临时的 col_buffer
        Tensor col_buffer({in_c * ksize * ksize, h_out * w_out}, input.device());
        
        forward_conv(const_cast<float*>(input.data()), 
                     output->data(), 
                     const_cast<float*>(weight.data()), 
                     col_buffer.data(), batch, in_c, out_c, h, w, ksize, stride, padding, get_cublas_handle());
        
        return output;
    }, py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("padding"));

    m.def("conv2d_backward", [](const Tensor& grad_output, const Tensor& input, const Tensor& weight, int stride, int padding) {
        int batch = input.shape()[0];
        int in_c = input.shape()[1];
        int h = input.shape()[2];
        int w = input.shape()[3];
        int out_c = weight.shape()[0];
        int ksize = weight.shape()[2];
        int h_out = grad_output.shape()[2];
        int w_out = grad_output.shape()[3];

        auto grad_input = std::make_shared<Tensor>(input.shape(), input.device());
        auto grad_weight = std::make_shared<Tensor>(weight.shape(), weight.device());
        
        Tensor col_buffer({in_c * ksize * ksize, h_out * w_out}, input.device());

        backward_conv(const_cast<float*>(input.data()), 
                      const_cast<float*>(grad_output.data()), 
                      const_cast<float*>(weight.data()),
                      col_buffer.data(), 
                      grad_input->data(), 
                      grad_weight->data(),
                      batch, in_c, out_c, h, w, ksize, stride, padding, get_cublas_handle());
                      
        return std::make_pair(grad_input, grad_weight);
    }, py::arg("grad_output"), py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("padding"));

    // --- Linear ---
    m.def("linear_forward", [](const Tensor& input, const Tensor& weight, const Tensor& bias) {
        int batch = input.shape()[0];
        int out_features = weight.shape()[0];
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_features}, input.device());
        linear_forward(input, weight, bias, *output); 
        return output;
    }, py::arg("input"), py::arg("weight"), py::arg("bias"));

    m.def("linear_backward", [](const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
        int batch = input.shape()[0];
        int in_features = input.shape()[1];
        int out_features = weight.shape()[0];
        auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{batch, in_features}, input.device());
        auto grad_weight = std::make_shared<Tensor>(std::vector<int64_t>{out_features, in_features}, input.device());
        auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, input.device());
        
        linear_backward(input, weight, grad_output, *grad_input, *grad_weight, *grad_bias);
        return std::make_tuple(grad_input, grad_weight, grad_bias);
    });

    // --- Activation / Elemwise ---
    m.def("relu_forward", [](const Tensor& x){ 
        auto y = std::make_shared<Tensor>(x.shape(), x.device()); 
        relu_forward(x, *y); 
        return y; 
    });
    
    m.def("relu_backward", [](const Tensor& g, const Tensor& x){ 
        auto gx = std::make_shared<Tensor>(x.shape(), x.device()); 
        relu_backward(g, x, *gx); 
        return gx; 
    });

    m.def("elementwise_add", [](const Tensor& a, const Tensor& b){ 
        auto c = std::make_shared<Tensor>(a.shape(), a.device()); 
        elementwise_add(a, b, *c); 
        return c; 
    });

    // --- Cross Entropy ---
    m.def("cross_entropy_loss", [](const Tensor& logits, const Tensor& labels) {
        Tensor probs(logits.shape(), logits.device());
        softmax_forward(logits, probs); 
        return cross_entropy_loss(probs, labels); 
    });

     m.def("cross_entropy_backward", [](const Tensor& logits, const Tensor& labels) {
        Tensor probs(logits.shape(), logits.device());
        softmax_forward(logits, probs);
        
        auto grad_input = std::make_shared<Tensor>(logits.shape(), logits.device());
        cross_entropy_backward(probs, labels, *grad_input);
        return grad_input;
    }, py::arg("logits"), py::arg("labels"));


    // --- MaxPool ---
    m.def("maxpool_forward", [](const Tensor& input, int ksize, int stride) {
        int batch = input.shape()[0];
        int c = input.shape()[1];
        int h = input.shape()[2];
        int w = input.shape()[3];
        int h_out = (h - ksize) / stride + 1;
        int w_out = (w - ksize) / stride + 1;
        
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, c, h_out, w_out}, input.device());
        auto mask = std::make_shared<Tensor>(std::vector<int64_t>{batch, c, h_out, w_out}, input.device());
        
        maxpool_forward(input, *output, *mask, ksize, stride, 0);
        return std::make_pair(output, mask);
    }, py::arg("input"), py::arg("ksize"), py::arg("stride"));

    m.def("maxpool_backward", [](const Tensor& grad_output, const Tensor& mask, int ksize, int stride) {
        int batch = mask.shape()[0];
        int c = mask.shape()[1];
        int h_out = mask.shape()[2];
        int w_out = mask.shape()[3];
        
        int h_in = (h_out - 1) * stride + ksize;
        int w_in = (w_out - 1) * stride + ksize;
        
        auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{batch, c, h_in, w_in}, mask.device());
        maxpool_backward(grad_output, mask, *grad_input, ksize, stride, 0);
        return grad_input;
    }, py::arg("grad_output"), py::arg("mask"), py::arg("ksize"), py::arg("stride"));
}