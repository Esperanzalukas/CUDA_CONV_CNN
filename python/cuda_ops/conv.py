import numpy as np
import my_deep_lib as cuda
from .module import Module, Parameter
from core.basic_operator import Op, Value

class Conv2DOp(Op):
    def __init__(self, weight, stride, padding):
        self.weight = weight
        self.stride = stride
        self.padding = padding
    
    def compute(self, x):
        if isinstance(x, np.ndarray):
            self.x = cuda.Tensor.from_numpy(x, cuda.Device.gpu(0))
        else:
            self.x = x

        w_data = self.weight.realize_cached_data()
        if isinstance(w_data, np.ndarray):
            self.w = cuda.Tensor.from_numpy(w_data, cuda.Device.gpu(0))
        else:
            self.w = w_data

        return cuda.conv2d_forward(self.x, self.w, self.stride, self.padding)
    
    def gradient(self, out_grad, node):
        grad_out = out_grad.realize_cached_data()
        if isinstance(grad_out, np.ndarray):
            grad_out = cuda.Tensor.from_numpy(grad_out, cuda.Device.gpu(0))
            
        # 修正：移除所有手动计算的 batch, h, w 等参数
        # 只传：input, grad_output, weight, stride, padding
        # C++ 端会自动从 Tensor.shape() 获取维度
        dx, dw = cuda.conv2d_backward(self.x, grad_out, self.w, self.stride, self.padding)
        
        gx = Value()
        gx._init(None, [], cached_data=dx)
        
        if self.weight.grad is None:
            self.weight.grad = Value()
            self.weight.grad._init(None, [], cached_data=dw)
        else:
            # 简单覆盖 (实际应累加，但暂且这样)
            self.weight.grad.cached_data = dw 

        return (gx,)

class Conv2D(Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (in_c * kernel_size * kernel_size))
        self.weight = Parameter((np.random.randn(out_c, in_c, kernel_size, kernel_size) * scale).astype(np.float32))
    
    def forward(self, x):
        return Conv2DOp(self.weight, self.stride, self.padding)(x)