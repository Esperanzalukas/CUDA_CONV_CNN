import numpy as np
import my_deep_lib as cuda

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = {}
        self.v = {}
        
        # 初始化动量缓存 (懒加载，这里只占位)
        # 注意：不要在这里初始化 Numpy 数组，因为那时我们还不知道参数的 shape
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # --- 1. 获取数据 & 准备 Numpy 环境 ---
            p_data_obj = p.realize_cached_data()
            g_data_obj = p.grad.realize_cached_data()
            
            is_gpu = False
            device = None

            if isinstance(p_data_obj, cuda.Tensor):
                w_np = p_data_obj.to_numpy()
                device = p_data_obj.device()
                is_gpu = True
            else:
                w_np = p_data_obj

            if isinstance(g_data_obj, cuda.Tensor):
                g_np = g_data_obj.to_numpy() # 拉回 CPU
            else:
                g_np = g_data_obj
            
            # 初始化状态 (如果还没有)
            if i not in self.m:
                self.m[i] = np.zeros_like(w_np)
                self.v[i] = np.zeros_like(w_np)
            
            # --- 2. Adam 核心逻辑 (Numpy) ---
            m = self.m[i]
            v = self.v[i]
            grad = g_np

            # Weight Decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * w_np

            # M = b1 * M + (1-b1) * G
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            
            # V = b2 * V + (1-b2) * G^2
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # Bias Correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update: W = W - lr * M_hat / (sqrt(V_hat) + eps)
            w_np -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # --- 3. 写回 ---
            if is_gpu:
                # 再次强调：这里依赖你 pybind_wrapper.cpp 里的 check_cuda 和 sync
                p.cached_data = cuda.Tensor.from_numpy(w_np, device)
            else:
                p.cached_data = w_np