import numpy as np
import my_deep_lib as cuda

class SGD:
    def __init__(self, params, lr=0.04, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {} 

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # 1. 准备数据
            if hasattr(p, "realize_cached_data"):
                p_data_obj = p.realize_cached_data()
            else:
                p_data_obj = p

            if hasattr(p.grad, "realize_cached_data"):
                g_data_obj = p.grad.realize_cached_data()
            else:
                g_data_obj = p.grad

            # 2. 拉回 CPU Numpy (使用 copy 确保内存独立)
            device = None
            is_gpu = False
            
            if isinstance(p_data_obj, cuda.Tensor):
                w_np = np.array(p_data_obj.to_numpy(), copy=True)
                device = p_data_obj.device()
                is_gpu = True
            else:
                w_np = p_data_obj

            if isinstance(g_data_obj, cuda.Tensor):
                g_np = np.array(g_data_obj.to_numpy(), copy=True)
            else:
                g_np = g_data_obj

            # 3. 【防爆核心】梯度裁剪 (Gradient Clipping)
            # 即使 Loss 已经归一化，卷积层的梯度因为空间累加，依然可能很大。
            # 这是训练 CNN 最稳健的技巧。
            grad_norm = np.linalg.norm(g_np)
            if grad_norm > 60.0:
                g_np = g_np * (60.0 / grad_norm)
            
            # 4. Numpy 更新
            if self.momentum > 0:
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(w_np)
                
                v = self.velocities[i]
                v[:] = self.momentum * v + g_np
                w_np -= self.lr * v
            else:
                w_np -= self.lr * g_np

            # 5. 写回GPU (防垃圾数据)
            if is_gpu:
                # 这一步至关重要，解决 garbage on HOST 问题
                w_final = np.ascontiguousarray(w_np.astype(np.float32))
                
                # 双重保险：再次检查是否有 NaN 或 无效大数
                if not np.isfinite(w_final).all():
                    print(f"[SGD Error] Parameter {i} contains NaNs after update! Resetting to random.")
                    w_final = np.random.randn(*w_final.shape).astype(np.float32) * 0.01

                p.cached_data = cuda.Tensor.from_numpy(w_final, device)
            else:
                p.cached_data = w_np