import sys
import os
import time
import numpy as np

# 1. 确保 Python 路径能找到 nn, optim 等包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

# 2. 引入你的深度学习库
import my_deep_lib as cuda
from core.basic_operator import Value
from nn import Module, Linear, ReLU, cross_entropy_loss
from optim import SGD
from dataloader import DataLoader, CIFAR10Dataset

# === [关键组件] 稳健的 Flatten ===
class Flatten:
    def __call__(self, x):
        d = x.realize_cached_data()
        
        # 1. 获取 Shape (兼容属性和方法)
        raw_shape = d.shape
        if callable(raw_shape):
            shape = raw_shape() # C++ method style
        else:
            shape = raw_shape   # Numpy style 
            
        # 2. 计算展平后的维度
        batch_size = shape[0]
        flatten_dim = 1
        for s in shape[1:]:
            flatten_dim *= s
            
        # 3. 执行 Reshape (所有操作拉回 Numpy 确保安全)
        if isinstance(d, cuda.Tensor):
            # C++ Tensor -> Numpy
            np_data = d.to_numpy()
            target_device = d.device()
        else:
            # Numpy
            np_data = d
            target_device = None
            
        # 强制 Numpy flatten
        reshaped_np = np_data.reshape(batch_size, flatten_dim).astype(np.float32)
        
        # 如果原来是 GPU，送回去
        if target_device is not None:
            # 这里的 ascontiguousarray 至关重要！
            reshaped_np = np.ascontiguousarray(reshaped_np)
            ret_data = cuda.Tensor.from_numpy(reshaped_np, target_device)
        else:
            ret_data = reshaped_np

        # 4. 包装成 Value
        v = Value()
        v._init(None, [x], cached_data=ret_data)
        
        # 5. 闭包捕获 shape 用于反向传播
        def backward_fn(out_grad, node):
            grad_d = out_grad.realize_cached_data()
            
            # 同样拉回 Numpy 做 reshape
            if isinstance(grad_d, cuda.Tensor):
                np_grad = grad_d.to_numpy()
                target_dev = grad_d.device()
            else:
                np_grad = grad_d
                target_dev = None
                
            # Reshape 回 [N, C, H, W]
            reshaped_grad = np_grad.reshape(shape).astype(np.float32)
            reshaped_grad = np.ascontiguousarray(reshaped_grad) # 确保连续
            
            if target_dev is not None:
                g_ret = cuda.Tensor.from_numpy(reshaped_grad, target_dev)
            else:
                g_ret = reshaped_grad
                
            gv = Value()
            gv._init(None, [], cached_data=g_ret)
            return (gv,)
            
        v._init(backward_fn, [x], cached_data=ret_data)
        return v

flatten = Flatten()

# === [关键修改] 使用 Pure MLP 进行冒烟测试 ===
# 一旦这个跑通 Acc > 40%，就可以确信框架基建全部正常
# 之后再把 CNN 换回来排查 Conv2d 即可
class MLPNet(Module):
    def __init__(self):
        super().__init__()
        # Input: 3 x 32 x 32 = 3072
        # Simple 2-layer MLP
        self.fc1 = Linear(3072, 256)
        self.relu = ReLU()
        self.fc2 = Linear(256, 10)
    
    def forward(self, x):
        x = flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def main():
    print("=== Training Start (MLP Mode) ===")
    
    # 1. 准备数据
    data_dir = './data'
    if not os.path.exists(data_dir):
        print("Dataset not found, downloading...")
        import torchvision
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
        
    train_ds = CIFAR10Dataset(data_dir, train=True)
    test_ds = CIFAR10Dataset(data_dir, train=False)
    
    # Batch Size 64
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 2. 初始化模型
    model = MLPNet()
    
    # 3. 初始化优化器
    # MLP 需要较大的 LR 才能起步，0.1 是个好起点
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    print(f"Training on {len(train_ds)} samples...")
    
    # 4. 训练循环
    for epoch in range(10): # 跑 10 轮先看效果
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i, (data, labels_idx) in enumerate(train_loader):
            # 准备数据：Numpy -> GPU Tensor -> Value
            if isinstance(data, np.ndarray):
                # 显式拷贝到 GPU
                gpu_tensor = cuda.Tensor.from_numpy(data.astype(np.float32), cuda.Device.gpu(0))
                x = Value()
                x._init(None, [], cached_data=gpu_tensor)
            else:
                x = Value(data) # 如果它已经是 Tensor
            
            # Label
            # 注意：Loss 函数内部应该能处理 int array 或 tensor，这里尽量保持原样
            
            # Forward
            logits = model(x)
            
            # Loss
            loss = cross_entropy_loss(logits, labels_idx)
            
            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # --- 统计 ---
            # 1. Loss Scalar
            l_val = loss.realize_cached_data()
            # 统一拉回 CPU 转 float
            if isinstance(l_val, cuda.Tensor):
                l_np = l_val.to_numpy()
                loss_scalar = float(l_np.item()) if l_np.size == 1 else float(l_np.mean()) # 防御性编程
            else:
                loss_scalar = float(l_val) if np.ndim(l_val)==0 else float(np.mean(l_val))
            
            total_loss += loss_scalar
            
            # 2. Acc
            logits_val = logits.realize_cached_data()
            if isinstance(logits_val, cuda.Tensor):
                logits_np = logits_val.to_numpy()
            else:
                logits_np = logits_val
            
            pred = np.argmax(logits_np, axis=1)
            correct += (pred == labels_idx).sum()
            total += labels_idx.shape[0]
            
            if (i+1) % 50 == 0:
                print(f"  [Epoch {epoch+1}] Batch {i+1}/{len(train_loader)}, Loss: {loss_scalar:.4f}, Acc: {100*correct/total:.2f}%")
        
        # Test Loop (可选，为节约时间可以先跳过或只跑部分)
        test_correct = 0
        test_total = 0
        # 简单取前 10 个 batch 测一下
        for j, (t_data, t_labels) in enumerate(test_loader):
            if j > 20: break 
            if isinstance(t_data, np.ndarray):
                gpu_tensor = cuda.Tensor.from_numpy(t_data.astype(np.float32), cuda.Device.gpu(0))
                tx = Value()
                tx._init(None, [], cached_data=gpu_tensor)
            else: tx = Value(t_data)
            
            t_logits = model(tx).realize_cached_data()
            if isinstance(t_logits, cuda.Tensor): t_logits = t_logits.to_numpy()
            t_pred = np.argmax(t_logits, axis=1)
            test_correct += (t_pred == t_labels).sum()
            test_total += t_labels.shape[0]

        train_acc = 100 * correct / total
        test_acc = 100 * test_correct / test_total
        print(f"=== Epoch {epoch+1} Done. Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, Test Acc (~2k)={test_acc:.2f}% ===")

if __name__ == "__main__":
    main()