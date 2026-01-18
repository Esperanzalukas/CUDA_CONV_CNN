import numpy as np
import my_deep_lib as cuda
from basic_operator import Op, Value, as_value
from .module import Module, Parameter

class Conv2DOp(Op):
    def __init__(self, stride, padding, kernel_size):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
    
    def compute(self, x, w):
        # x: Tensor [N, C, H, W]
        # w: Tensor [Out, In, K, K]
        self.inputs = [x, w]
        
        if isinstance(x, cuda.Tensor):
             return cuda.conv2d_forward(x, w, self.stride, self.padding)
        else:
             raise RuntimeError("Conv2D currently only supports GPU Tensor inputs.")
    
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        w = node.inputs[1]
        
        grad_out = out_grad.realize_cached_data()
        x_data = x.realize_cached_data()
        w_data = w.realize_cached_data()
        
        dx, dw = cuda.conv2d_backward(grad_out, x_data, w_data, self.stride, self.padding)
        return as_value(dx), as_value(dw)

# ... existing imports ...

class Conv2D(Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, bias=True): # 增加 bias 参数
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # === 1. Kaiming Init ===
        scale = np.sqrt(4.0 / (in_c * kernel_size * kernel_size))
        
        # 产生随机数
        w_tmp = np.random.randn(out_c, in_c, kernel_size, kernel_size) * scale
        
        # 【至关重要】强制转换为 float32 且 内存连续(C-contiguous)
        # 很多垃圾数据的 bug 都是因为传了非连续内存给 C++ memcpy
        w_np = np.ascontiguousarray(w_tmp.astype(np.float32))

        print(f"[Python Init] Host Data First Element: {w_np.flatten()[0]}")
        
        # === 2. 传输到 GPU & 立即校验 ===
        if hasattr(cuda.Tensor, "from_numpy"):
            w_tensor = cuda.Tensor.from_numpy(w_np, cuda.Device.gpu(0))
            
            w_check = w_tensor.to_numpy() # 读回来
            diff = np.abs(w_check - w_np).mean()
            print(f"[Conv Init] GPU Sync Check: Diff={diff:.6f}, MeanAbs_GPU={np.mean(np.abs(w_check)):.6f}")
            
            if diff > 1e-4:
                # 如果这里报错，说明 C++ 的 cudaMemcpy 有问题
                raise RuntimeError(f"GPU Memory Initialization FAILED! Diff={diff}.")
            
            # 如果上面没报错，但训练时权重变成了 10^22
            # 说明 Parameter 类或者 SGD 更新逻辑把权重搞坏了
            self.weight = Parameter(w_tensor)
        else:
            self.weight = Parameter(w_np)

         # === Bias Init ===
        if bias:
            # bias 形状通常是 [Out_C]，但在做广播加法时，
            # 需要 reshape 成 [1, Out_C, 1, 1] 以便加到 (N, C, H, W) 上
            self.bias = Parameter(np.zeros((1, out_c, 1, 1), dtype=np.float32))
        else:
            self.bias = None
    
    # ... existing forward ...
    
    def forward(self, x):
        # 1. 卷积运算
        out = Conv2DOp(self.stride, self.padding, self.kernel_size)(x, self.weight)
        
        # 2. 加上 Bias (利用 Broadcast)
        if self.bias is not None:
            # 需要 basic_operator.py 里支持 tensor + tensor 的广播加法
            # 或者在这里手动广播。由于只有 Channel 维对应，通常 C++ elementwise add 需要支持 broadcasting
            # 这里为简单起见，假设 Tensor 实现了基础的广播或者我们暂且在 Python 端看起来像个 op
            out = out + self.bias
        return out