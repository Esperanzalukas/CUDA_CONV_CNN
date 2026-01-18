import numpy as np
import my_deep_lib as cuda
from basic_operator import Op, Value, as_value
from .module import Module, Parameter

class LinearOp(Op):
    def compute(self, x, w, b):
        # === 临时强制 CPU fallback 以规避 C++ SGEMM/Memory Error ===
        # 如果输入是 C++ Tensor，转回 Numpy 计算
        if isinstance(x, cuda.Tensor):
            x_np = x.to_numpy()
            w_np = w.to_numpy()
            b_np = b.to_numpy()
            
            # Numpy MatMul: (Batch, In) @ (In, Out) + (Out)
            # 此时我们的权重 w 是 [In, Out] 形状
            out_np = x_np @ w_np + b_np
            
            # 结果转回 C++ Tensor (GPU)
            # 注意：新 Tensor 需要在正确的 device 上
            return cuda.Tensor.from_numpy(out_np.astype(np.float32), x.device())
            
        # 纯 CPU 模式
        return x @ w + b

    def gradient(self, out_grad, node):
        x, w, b = node.inputs
        
        grad_out = out_grad.realize_cached_data()
        x_data = x.realize_cached_data()
        w_data = w.realize_cached_data()
        
        # === 临时强制 CPU fallback ===
        if isinstance(grad_out, cuda.Tensor):
            # 将所有数据转回 CPU
            dout_np = grad_out.to_numpy()
            x_np = x_data.to_numpy()
            w_np = w_data.to_numpy()
            
            # Numpy Backward
            # Y = X @ W + B
            # dL/dX = dL/dY @ W.T
            # dL/dW = X.T @ dL/dY
            # dL/dB = sum(dL/dY, axis=0)
            
            dx_np = dout_np @ w_np.T
            dw_np = x_np.T @ dout_np
            db_np = np.sum(dout_np, axis=0)
            
            # 转回 GPU
            dev = grad_out.device()
            dx = cuda.Tensor.from_numpy(dx_np.astype(np.float32), dev)
            dw = cuda.Tensor.from_numpy(dw_np.astype(np.float32), dev)
            db = cuda.Tensor.from_numpy(db_np.astype(np.float32), dev)
            
        else:
             dx = grad_out @ w_data.T
             dw = x_data.T @ grad_out
             db = np.sum(grad_out, axis=0)
             
        return as_value(dx), as_value(dw), as_value(db)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 使用 [In, Out] 形状，这最符合 Numpy 的 X @ W 习惯
        limit = np.sqrt(6 / (in_features + out_features))
        w_data = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        
        if hasattr(cuda.Tensor, "from_numpy"):
             w_tensor = cuda.Tensor.from_numpy(w_data, cuda.Device.gpu(0))
             b_tensor = cuda.Tensor.from_numpy(np.zeros(out_features, dtype=np.float32), cuda.Device.gpu(0))
             self.weight = Parameter(w_tensor)
             self.bias = Parameter(b_tensor)
        else:
             self.weight = Parameter(w_data)
             self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        
    def forward(self, x):
        return LinearOp()(x, self.weight, self.bias)