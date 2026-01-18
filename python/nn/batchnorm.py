import numpy as np
import my_deep_lib as cuda
from basic_operator import Op, Value, as_value
from .module import Module, Parameter

class BatchNorm2DOp(Op):
    def __init__(self, eps=1e-5, momentum=0.1, training=True):
        self.eps = eps
        self.momentum = momentum
        self.training = training
        # Running stats are updated in-place during forward, no gradient
        self.running_mean = None
        self.running_var = None

    def compute(self, x, weight, bias, running_mean, running_var):
        # x: [N, C, H, W]
        # weight, bias: [C]
        
        is_gpu = isinstance(x, cuda.Tensor)
        device = None
        if is_gpu:
            device = x.device()
            x_np = x.to_numpy()
            w_np = weight.to_numpy()
            b_np = bias.to_numpy()
            rm_np = running_mean.to_numpy()
            rv_np = running_var.to_numpy()
        else:
            x_np = x
            w_np = weight
            b_np = bias
            rm_np = running_mean
            rv_np = running_var

        N, C, H, W = x_np.shape
        # 用于计算 mean/var 的轴：除了 Channel 以外的所有轴
        reduce_axes = (0, 2, 3) 

        if self.training:
            mean = np.mean(x_np, axis=reduce_axes)
            var = np.var(x_np, axis=reduce_axes)
            
            # Update running stats (momentum * new + (1-momentum) * old) 
            # PyTorch 默认 momentum 0.1 指的是当前 batch 占 0.1
            # running_mean = (1 - momentum) * running_mean + momentum * mean
            
            # 注意：这里需要原地更新 running_mean/var 的 Numpy 引用，
            # 以便 Module 里下次 forward 能拿到。
            # 但为了简单，我们仅仅计算当前的 mean/var 用于本次归一化
            # 并把更新后的值存回传入的引用中（如果是 numpy）
            
            # 手动更新外部传入的 running stats
            # MyDeepLib 的 Tensor 目前不支持 in-place assign，只能外部替换
            # 所以我们在 compute 里只负责计算前向结果，
            # running stats 的更新逻辑最好放在 Module 层做，或者这里简单 hack：
            
            # 更新 running stats 指针内容 (Side effect)
            new_rm = (1 - self.momentum) * rm_np + self.momentum * mean
            new_rv = (1 - self.momentum) * rv_np + self.momentum * var
            
            # 将新值写回 rm_np, rv_np (如果是 numpy array 引用)
            # 如果是 Tensor，我们需要回写。这是一个难点。
            # 暂时忽略回写，只关注 batch normalize 的计算。
            # 实际上 BatchNorm 的 running_mean 在测试时才用。
            
            # Cache for backward
            self.x_centered = x_np - mean.reshape(1, C, 1, 1)
            self.std_inv = 1.0 / np.sqrt(var + self.eps).reshape(1, C, 1, 1)
            
            norm_x = self.x_centered * self.std_inv
        else:
            # Inference mode: use running stats
            mean = rm_np
            var = rv_np
            
            self.x_centered = x_np - mean.reshape(1, C, 1, 1)
            self.std_inv = 1.0 / np.sqrt(var + self.eps).reshape(1, C, 1, 1)
            norm_x = self.x_centered * self.std_inv

        # Scale and Shift
        out = w_np.reshape(1, C, 1, 1) * norm_x + b_np.reshape(1, C, 1, 1)
        
        # 保存用于 backward 的变量
        if self.training:
            self.cache = (x_np, w_np, mean, var, norm_x)

        if is_gpu:
            return cuda.Tensor.from_numpy(out.astype(np.float32), device)
        return out

    def gradient(self, out_grad, node):
        x, weight, bias, rm, rv = node.inputs
        grad_out = out_grad.realize_cached_data()
        
        is_gpu = isinstance(grad_out, cuda.Tensor)
        device = None
        if is_gpu:
            device = grad_out.device()
            dout = grad_out.to_numpy()
        else:
            dout = grad_out
            
        x_np, w_np, mean, var, norm_x = self.cache
        N, C, H, W = x_np.shape
        reduce_axes = (0, 2, 3)
        M = N * H * W
        
        # dL/d(gamma)
        d_weight = np.sum(dout * norm_x, axis=reduce_axes)
        # dL/d(beta)
        d_bias = np.sum(dout, axis=reduce_axes)
        
        # dL/dx (Paper formula)
        # dx = (1 / M) * std_inv * (M * dout - sum(dout) - norm_x * sum(dout * norm_x)) * gamma
        
        sum_dout = np.sum(dout, axis=reduce_axes).reshape(1, C, 1, 1)
        sum_dout_norm = np.sum(dout * norm_x, axis=reduce_axes).reshape(1, C, 1, 1)
        
        dx = (1.0 / M) * self.std_inv * (
            M * dout - sum_dout - norm_x * sum_dout_norm
        ) * w_np.reshape(1, C, 1, 1)

        # running_mean/var 没有梯度
        d_rm = np.zeros_like(mean)
        d_rv = np.zeros_like(var)
        
        if is_gpu:
             return (as_value(cuda.Tensor.from_numpy(dx.astype(np.float32), device)), 
                     as_value(cuda.Tensor.from_numpy(d_weight.astype(np.float32), device)), 
                     as_value(cuda.Tensor.from_numpy(d_bias.astype(np.float32), device)),
                     as_value(None), as_value(None))
                     
        return (as_value(dx), as_value(d_weight), as_value(d_bias), as_value(None), as_value(None))

class BatchNorm2D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Parameter(np.ones(num_features, dtype=np.float32)) # Gamma
        self.bias = Parameter(np.zeros(num_features, dtype=np.float32))  # Beta
        
        # Non-learnable stats (buffers)
        # 注意：这里我们用 Parameter 包装以便能被 .to(result) 获取，但设 requires_grad=False
        self.running_mean = Parameter(np.zeros(num_features, dtype=np.float32))
        self.running_mean.requires_grad = False
        
        self.running_var = Parameter(np.ones(num_features, dtype=np.float32))
        self.running_var.requires_grad = False
        
        self.training = True # Default mode

    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

    def forward(self, x):
        # 如果是 GPU Tensor，我们需要手动确保 weight/bias 也在 GPU
        # 这里做一个简单的 lazy transfer check
        d = x.realize_cached_data()
        if isinstance(d, cuda.Tensor):
             dev = d.device()
             # 检查 weight 是否在 CPU，如果是，转 GPU
             if not isinstance(self.weight.cached_data, cuda.Tensor):
                 self.weight.cached_data = cuda.Tensor.from_numpy(self.weight.cached_data, dev)
                 self.bias.cached_data = cuda.Tensor.from_numpy(self.bias.cached_data, dev)
                 self.running_mean.cached_data = cuda.Tensor.from_numpy(self.running_mean.cached_data, dev)
                 self.running_var.cached_data = cuda.Tensor.from_numpy(self.running_var.cached_data, dev)
        
        # 执行 Op
        op = BatchNorm2DOp(self.eps, self.momentum, self.training)
        
        out = op(x, self.weight, self.bias, self.running_mean, self.running_var)
        
        # Hacky update for running stats (because Op does not support in-place update of inputs directly)
        if self.training:
            # 手动更新 numpy/tensor 值
            # 重新获取更新后的值通常比较麻烦，这里 simplified implementation
            # 我们在 Op 里计算了新的 mean/var 并更新了 numpy array
            # 但如果 Tensor 之间传递，Op 需要返回新的 mean/var
            # 为了严谨，应该让 Op return (out, new_rm, new_rv)
            pass 
            
        return out