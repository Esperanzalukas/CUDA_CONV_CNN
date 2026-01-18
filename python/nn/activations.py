import numpy as np
import my_deep_lib as cuda
from .module import Module
from core.basic_operator import Op, Value, as_value

# === ReLU ===
class ReLUOp(Op):
    def compute(self, x):
        self.x = x 
        return cuda.relu_forward(x)
    
    def gradient(self, out_grad, node):
        grad_out = out_grad.realize_cached_data()
        
        if isinstance(grad_out, np.ndarray):
             # ...existing code...
             # 注意：确保 device 一致
             if isinstance(self.x, cuda.Tensor):
                grad_out = cuda.Tensor.from_numpy(grad_out, self.x.device())
             else:
                pass # CPU 情况
            
        # === fix start ===
        # C++ signature: relu_backward(input_x, grad_output)
        # Python previous: relu_backward(grad_out, self.x) -> ERROR (reversed)
        grad_x = cuda.relu_backward(self.x, grad_out) 
        # === fix end ===
        
        return (as_value(grad_x),)

class ReLU(Module):
    def forward(self, x):
        return ReLUOp()(x)

# === Sigmoid ===
class SigmoidOp(Op):
    def compute(self, x):
        # x 是 Tensor
        # 保存 output (y) 用于反向传播，因为 Sigmoid 定义:
        # gradient = grad_out * y * (1 - y)
        self.out = cuda.sigmoid_forward(x)
        return self.out
    
    def gradient(self, out_grad, node):
        grad_out = out_grad.realize_cached_data()
        
        if isinstance(grad_out, np.ndarray):
            grad_out = cuda.Tensor.from_numpy(grad_out, cuda.Device.gpu(0))
            
        # 假设 C++ 接口: sigmoid_backward(grad_output, output_y) -> grad_input
        grad_x = cuda.sigmoid_backward(grad_out, self.out)
        
        return (as_value(grad_x),)

class Sigmoid(Module):
    def forward(self, x):
        return SigmoidOp()(x)

# === Softmax ===
# 通常 Softmax 和 CrossEntropy 结合在一起用 (LogSoftmax)，数值更稳定
# 但如果需要单独用的 Softmax
class SoftmaxOp(Op):
    def compute(self, x):
        # x: Tensor
        self.out = cuda.softmax_forward(x)
        return self.out
    
    def gradient(self, out_grad, node):
        # 单独的 Softmax 梯度比较复杂，且 C++ 库目前通常没有单独实现 softmax_backward
        # 大多数情况下都在 CrossEntropyLoss 里算掉了。
        # 如果必须要有，可以暂时回退到 Numpy
        
        dout = out_grad.realize_cached_data()
        out = self.out
        
        is_gpu = isinstance(dout, cuda.Tensor)
        if is_gpu:
            dout_np = dout.to_numpy()
            out_np = out.to_numpy()
        else:
            dout_np = dout
            out_np = out
            
        # Softmax Gradient:
        # dX_i = Y_i * (dL/dY_i - sum_k(dL/dY_k * Y_k))
        sum_term = np.sum(dout_np * out_np, axis=1, keepdims=True)
        dx_np = out_np * (dout_np - sum_term)
        
        if is_gpu:
            return (as_value(cuda.Tensor.from_numpy(dx_np.astype(np.float32), dout.device())),)
        return (as_value(dx_np),)

class Softmax(Module):
    def forward(self, x):
        return SoftmaxOp()(x)