import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# 引入 C++ 扩展库
import my_deep_lib 

from basic_operator import Value
from nn import Module, Parameter, Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss
from optim import SGD
from data import DataLoader, CIFAR10Dataset

class Flatten:
    def __call__(self, x):
        d = x.realize_cached_data()
        
        # === 终极修复：获取 shape ===
        # 1. 尝试直接获取属性 (Numpy 或 Property-based Tensor)
        shape_attr = getattr(d, "shape", None)
        
        if isinstance(shape_attr, (list, tuple)):
            self.shape = shape_attr
        # 2. 如果属性是 method (Method-based Tensor)，则调用它
        elif callable(shape_attr):
            self.shape = shape_attr()
        else:
            raise RuntimeError(f"Unknown shape type for data: {type(d)}")
            
        # 获取 numel
        if hasattr(d, "numel"):
             numel = d.numel()
        else:
             numel = d.size
        
        current_batch = self.shape[0]
        flatten_dim = numel // current_batch
        
        # DEBUG:
        if not hasattr(self, "printed"):
            print(f"DEBUG: Flatten shape: {self.shape} -> [Batch={current_batch}, Dim={flatten_dim}]")
            self.printed = True
        
        # 执行 Reshape
        if isinstance(d, my_deep_lib.Tensor):
            if numel % current_batch != 0:
                 raise RuntimeError(f"Flatten error: numel {numel} not divisible by batch {current_batch}")
            # C++ Tensor reshape(list)
            new_d = d.reshape([current_batch, flatten_dim]) 
        else:
            # Numpy
            new_d = d.reshape(current_batch, -1).astype(np.float32)
            
        r = Value()
        r._init(None, [x], cached_data=new_d)
        r.op = self
        return r

    def gradient(self, out_grad, node):
        g = out_grad.realize_cached_data()
        
        if isinstance(g, my_deep_lib.Tensor):
            target_shape = list(self.shape)
            dg = g.reshape(target_shape)
        else:
            dg = g.reshape(self.shape).astype(np.float32)
            
        gv = Value()
        gv._init(None, [], cached_data=dg)
        return (gv,)

flatten = Flatten()

class CNN(Module):
    def __init__(self):
        super().__init__()
        # Conv1: 3 -> 32
        self.conv1 = Conv2D(3, 32, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)
        
        # Conv2: 32 -> 64
        self.conv2 = Conv2D(32, 64, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)
        
        # Flatten 后尺寸: 64通道 * 8 * 8
        self.fc1 = Linear(64*8*8, 256)
        self.relu3 = ReLU()
        self.fc2 = Linear(256, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

def main():
    print("Loading data...")
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    train_ds = CIFAR10Dataset('./data', train=True)
    test_ds = CIFAR10Dataset('./data', train=False)
    
    # 稍微调小 batch_size
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = CNN()
    
    # 计算参数量
    total_params = 0
    for p in model.parameters():
        d = p.cached_data
        if hasattr(d, "numel"):
            total_params += d.numel()
        else:
            total_params += d.size
    print(f"Parameters: {total_params:,}")
    
    opt = SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("Training...")
    
    for epoch in range(10):
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        
        for i, (data, labels) in enumerate(train_loader):

            if data.shape[-1] == 3: # 检测是否为 Channel Last
                data = data.transpose(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
    

            # Numpy -> GPU Tensor
            gpu_data = my_deep_lib.Tensor.from_numpy(data, my_deep_lib.Device.gpu(0))
            gpu_labels = my_deep_lib.Tensor.from_numpy(labels.astype(np.float32), my_deep_lib.Device.gpu(0))

            x = Value(); x._init(None, [], cached_data=gpu_data)
            y = Value(); y._init(None, [], cached_data=gpu_labels, requires_grad=False)
            
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_val = loss.realize_cached_data()
            if hasattr(loss_val, "to_numpy"):
                loss_scalar = float(loss_val.to_numpy())
            else:
                loss_scalar = float(loss_val)
            total_loss += loss_scalar
            
            logits_data = logits.realize_cached_data()
            if hasattr(logits_data, "to_numpy"):
                logits_np = logits_data.to_numpy()
                labels_np = labels 
            else:
                logits_np = logits_data
                labels_np = labels
                
            pred = logits_np.argmax(axis=1)
            correct += (pred == labels_np).sum()
            total += len(labels)
            
            if (i+1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {total_loss/(i+1):.4f}, Acc: {100*correct/total:.2f}%")
        
        test_correct, test_total = 0, 0
        for data, labels in test_loader:
            gpu_data = my_deep_lib.Tensor.from_numpy(data, my_deep_lib.Device.gpu(0))
            x = Value(); x._init(None, [], cached_data=gpu_data)
            
            logits = model(x)
            logits_data = logits.realize_cached_data()
            if hasattr(logits_data, "to_numpy"):
                logits_np = logits_data.to_numpy()
            else:
                logits_np = logits_data
                
            pred = logits_np.argmax(axis=1)
            test_correct += (pred == labels).sum()
            test_total += len(labels)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Train Acc={100*correct/total:.2f}%, Test Acc={100*test_correct/test_total:.2f}%, "
              f"Time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()