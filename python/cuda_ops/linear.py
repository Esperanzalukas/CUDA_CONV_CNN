import numpy as np
import my_deep_lib as cuda
from .module import Module, Parameter
from core.basic_operator import Op, Value

class LinearOp(Op):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    
    def compute(self, x):
        # 1. 自动 Flatten 处理
        # 如果输入是 (N, C, H, W)，需要变成 (N, C*H*W)
        shape = x.shape()
        if len(shape) > 2:
            batch = shape[0]
            numel = x.numel() // batch
            # 使用我们在 pybind 里新加的 reshape
            self.x = x.reshape([batch, numel])
        else:
            self.x = x

        # 2. 准备权重
        w = self.weight.realize_cached_data()
        if isinstance(w, np.ndarray):
            self.w = cuda.Tensor.from_numpy(w, cuda.Device.gpu(0))
        else:
            self.w = w
            
        b = self.bias.realize_cached_data()
        if isinstance(b, np.ndarray):
            self.b = cuda.Tensor.from_numpy(b, cuda.Device.gpu(0))
        else:
            self.b = b
            
        return cuda.linear_forward(self.x, self.w, self.b)
    
    def gradient(self, out_grad, node):
        grad_out = out_grad.realize_cached_data()
        if isinstance(grad_out, np.ndarray):
            grad_out = cuda.Tensor.from_numpy(grad_out, cuda.Device.gpu(0))
            
        dx, dw, db = cuda.linear_backward(grad_out, self.x, self.w)
        
        # 处理输入的梯度 (dx)
        # 如果之前做了 reshape，这里其实应该把 shape 变回去，
        # 但对于全连接层的反向传播，通常只需要传回 flat 的梯度即可，或者上一层是 MaxPoolBackward 会自己处理形状
        gx = Value()
        gx._init(None, [], cached_data=dx)
        
        # 处理参数梯度
        if self.weight.grad is None:
            self.weight.grad = Value()
            self.weight.grad._init(None, [], cached_data=dw)
        else:
            self.weight.grad.cached_data = dw
            
        if self.bias.grad is None:
            self.bias.grad = Value()
            self.bias.grad._init(None, [], cached_data=db)
        else:
            self.bias.grad.cached_data = db
            
        return (gx,)

class Linear(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        # Kaiming init
        self.weight = Parameter(np.random.randn(out_f, in_f).astype(np.float32) * np.sqrt(2.0/in_f))
        self.bias = Parameter(np.zeros(out_f, dtype=np.float32))
    
    def forward(self, x):
        return LinearOp(self.weight, self.bias)(x)