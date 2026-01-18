"""
激活函数的 CUDA 算子包装
"""
import numpy as np
from typing import Tuple
import my_deep_lib as cuda_lib
from core.basic_operator import Op, Value

class ReLUOp(Op):
    """ReLU 激活函数"""
    
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """
        前向传播：y = max(0, x)
        """
        # 转换为 CUDA Tensor
        cuda_input = cuda_lib.Tensor.from_numpy(input_data, cuda_lib.Device.gpu(0))
        cuda_output = cuda_lib.Tensor(list(input_data.shape), cuda_lib.Device.gpu(0))
        
        # 调用 CUDA kernel
        cuda_lib.relu_forward(cuda_input, cuda_output)
        
        return cuda_output.to_numpy()
    
    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value]:
        """
        反向传播：grad_input = grad_output * (input > 0)
        """
        input_node = node.inputs[0]
        input_data = input_node.realize_cached_data()
        grad_output_data = out_grad.realize_cached_data()
        
        # 转换为 CUDA Tensor
        cuda_input = cuda_lib.Tensor.from_numpy(input_data, cuda_lib.Device.gpu(0))
        cuda_grad_output = cuda_lib.Tensor.from_numpy(grad_output_data, cuda_lib.Device.gpu(0))
        cuda_grad_input = cuda_lib.Tensor(list(input_data.shape), cuda_lib.Device.gpu(0))
        
        # 调用 CUDA backward kernel
        cuda_lib.relu_backward(cuda_input, cuda_grad_output, cuda_grad_input)
        
        grad_input_np = cuda_grad_input.to_numpy()
        grad_input = Value.make_const(grad_input_np, requires_grad=False)
        
        return (grad_input,)

class SoftmaxOp(Op):
    """Softmax 激活函数（通常用于输出层）"""
    
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """
        前向传播：softmax(x) = exp(x) / sum(exp(x))
        input_data: [batch, num_classes]
        """
        # 转换为 CUDA Tensor
        cuda_input = cuda_lib.Tensor.from_numpy(input_data, cuda_lib.Device.gpu(0))
        cuda_output = cuda_lib.Tensor(list(input_data.shape), cuda_lib.Device.gpu(0))
        
        # 调用 CUDA kernel
        cuda_lib.softmax_forward(cuda_input, cuda_output)
        
        return cuda_output.to_numpy()
    
    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value]:
        """
        Softmax 的梯度比较复杂，但通常与 CrossEntropy 融合计算
        这里提供一个简化版本
        """
        # 获取 softmax 的输出
        output_data = node.realize_cached_data()
        grad_output_data = out_grad.realize_cached_data()
        
        # Softmax 梯度：grad_input = softmax * (grad_output - sum(grad_output * softmax))
        # 这里用 NumPy 实现（因为 CUDA kernel 里没有单独的 softmax backward）
        sum_term = np.sum(grad_output_data * output_data, axis=1, keepdims=True)
        grad_input_np = output_data * (grad_output_data - sum_term)
        
        grad_input = Value.make_const(grad_input_np, requires_grad=False)
        return (grad_input,)

class SigmoidOp(Op):
    """Sigmoid 激活函数"""
    
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """前向传播：y = 1 / (1 + exp(-x))"""
        cuda_input = cuda_lib.Tensor.from_numpy(input_data, cuda_lib.Device.gpu(0))
        cuda_output = cuda_lib.Tensor(list(input_data.shape), cuda_lib.Device.gpu(0))
        
        cuda_lib.sigmoid_forward(cuda_input, cuda_output)
        
        return cuda_output.to_numpy()
    
    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value]:
        """反向传播：grad_input = grad_output * y * (1 - y)"""
        output_data = node.realize_cached_data()  # sigmoid 的输出
        grad_output_data = out_grad.realize_cached_data()
        
        cuda_output = cuda_lib.Tensor.from_numpy(output_data, cuda_lib.Device.gpu(0))
        cuda_grad_output = cuda_lib.Tensor.from_numpy(grad_output_data, cuda_lib.Device.gpu(0))
        cuda_grad_input = cuda_lib.Tensor(list(output_data.shape), cuda_lib.Device.gpu(0))
        
        cuda_lib.sigmoid_backward(cuda_output, cuda_grad_output, cuda_grad_input)
        
        grad_input_np = cuda_grad_input.to_numpy()
        grad_input = Value.make_const(grad_input_np, requires_grad=False)
        
        return (grad_input,)

# 高层接口
def relu(x: Value) -> Value:
    """ReLU 激活函数"""
    return ReLUOp()(x)

def softmax(x: Value) -> Value:
    """Softmax 激活函数"""
    return SoftmaxOp()(x)

def sigmoid(x: Value) -> Value:
    """Sigmoid 激活函数"""
    return SigmoidOp()(x)