"""
CUDA MaxPool2D 算子的 Python 包装
"""
import numpy as np
from typing import Tuple
import my_deep_lib as cuda_lib
from basic_operator import Op, Value

class MaxPool2DOp(Op):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.mask = None  # 存储 forward 时的 mask，用于 backward
    
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """
        input_data: [batch, channels, height, width]
        """
        batch, channels, in_h, in_w = input_data.shape
        
        # 计算输出尺寸
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 转换为 CUDA Tensor
        cuda_input = cuda_lib.Tensor.from_numpy(input_data, cuda_lib.Device.gpu(0))
        cuda_output = cuda_lib.Tensor([batch, channels, out_h, out_w], cuda_lib.Device.gpu(0))
        cuda_mask = cuda_lib.Tensor([batch, channels, out_h, out_w], cuda_lib.Device.gpu(0))
        
        # 调用 CUDA kernel
        cuda_lib.maxpool_forward(cuda_input, cuda_output, cuda_mask,
                                self.kernel_size, self.stride, self.padding)
        
        # 保存 mask 用于 backward（转为 NumPy 保存）
        self.mask = cuda_mask.to_numpy()
        
        return cuda_output.to_numpy()
    
    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value]:
        """
        MaxPool 的梯度传播
        """
        input_node = node.inputs[0]
        input_data = input_node.realize_cached_data()
        grad_output_data = out_grad.realize_cached_data()
        
        # 转换为 CUDA Tensor
        cuda_grad_output = cuda_lib.Tensor.from_numpy(grad_output_data, cuda_lib.Device.gpu(0))
        cuda_mask = cuda_lib.Tensor.from_numpy(self.mask, cuda_lib.Device.gpu(0))
        cuda_grad_input = cuda_lib.Tensor(list(input_data.shape), cuda_lib.Device.gpu(0))
        
        # 调用 CUDA backward kernel
        cuda_lib.maxpool_backward(cuda_grad_output, cuda_mask, cuda_grad_input,
                                 self.kernel_size, self.stride, self.padding)
        
        grad_input_np = cuda_grad_input.to_numpy()
        grad_input = Value.make_const(grad_input_np, requires_grad=False)
        
        return (grad_input,)

def maxpool2d(input: Value, kernel_size, stride=None, padding=0) -> Value:
    """
    MaxPool2D 操作
    
    Args:
        input: [batch, channels, H, W]
        kernel_size: 池化窗口大小
        stride: 步长（默认等于 kernel_size）
        padding: 填充
    
    Returns:
        output: [batch, channels, H_out, W_out]
    """
    return MaxPool2DOp(kernel_size, stride, padding)(input)