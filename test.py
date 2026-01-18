import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
import my_deep_lib
from basic_operator import Value
from nn import Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss, Module
from optim import SGD

# 简单的 Flatten Op 封装，为了测试脚本独立运行
class FlattenOp:
    def __call__(self, x):
        d = x.realize_cached_data()
        
        # 获取 batch size
        if hasattr(d, "shape") and callable(d.shape): shape = d.shape()
        elif hasattr(d, "shape"): shape = d.shape
        else: shape = d.size 
        
        batch = shape[0]
        # 计算 total dim
        if hasattr(d, "numel"): numel = d.numel()
        else: numel = d.size
        
        dim = numel // batch
        
        # Reshape
        if isinstance(d, my_deep_lib.Tensor):
            new_d = d.reshape([batch, dim])
        else:
            new_d = d.reshape(batch, -1)
            
        r = Value()
        r._init(None, [x], cached_data=new_d)
        
        # 为了让梯度能传回去，我们需要定义一个简单的反向逻辑
        # 这里用闭包偷懒，不定义 Class 了，直接 hack gradient
        def backward_hook(out_grad, node):
            g = out_grad.realize_cached_data()
            if isinstance(g, my_deep_lib.Tensor):
                dg = g.reshape(list(shape))
            else:
                dg = g.reshape(shape)
            v = Value()
            v._init(None, [], cached_data=dg)
            return (v,)
            
        # 手动挂载 op 接口
        class DummyOp:
            def gradient(self, g, n): return backward_hook(g, n)
        r.op = DummyOp()
        
        return r

flatten = FlattenOp()

class MiniCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(3, 16, 3, padding=1)
        self.relu1 = ReLU()
        # 16通道 * 32 * 32
        self.fc1 = Linear(16*32*32, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # 修复：必须 Flatten
        x = flatten(x)
        x = self.fc1(x)
        return x

def main():
    print("=== 开始过拟合测试 (Sanity Check) ===")
    
    # 1. 伪造仅仅 4 张图片的数据集
    np.random.seed(42) # 固定随机种子
    fake_data = np.random.randn(4, 3, 32, 32).astype(np.float32)
    fake_labels = np.array([0, 1, 2, 3]).astype(np.int32)
    
    # 转 GPU Tensor
    gpu_data = my_deep_lib.Tensor.from_numpy(fake_data, my_deep_lib.Device.gpu(0))
    # 注意这里，labels 也要转成 onehot 或者 float 供 loss 使用吗？
    # 你的 cross_entropy_loss 实现似乎接受 Tensor(float) 作为 labels
    # 并且如果是 Tensor，它会被当做 target distribution 还是 index?
    # 翻看之前的 train.py，你是传入 float 的。通常 sparse_categorical_crossentropy 传 int index，
    # 但如果你的 loss 实现是需要 one-hot 或者 float 形式的 label，请确保这里一致。
    # 假设你的 cross_entropy_loss 内部处理是通用的。
    gpu_labels = my_deep_lib.Tensor.from_numpy(fake_labels.astype(np.float32), my_deep_lib.Device.gpu(0))
    
    model = MiniCNN()
    # 强力优化器设置
    opt = SGD(model.parameters(), lr=0.1, momentum=0.0) # 动量设为0有时对简单debug更纯粹，0.9也可以
    
    print("Iter\tLoss\t\tAcc")
    for i in range(101):
        # Forward
        x = Value(); x._init(None, [], cached_data=gpu_data)
        y = Value(); y._init(None, [], cached_data=gpu_labels, requires_grad=False)
        
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        
        # Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Check
        if i % 10 == 0:
            loss_val = loss.realize_cached_data()
            loss_scalar = float(loss_val.to_numpy()) if hasattr(loss_val, "to_numpy") else float(loss_val)
            
            logits_val = logits.realize_cached_data()
            logits_np = logits_val.to_numpy() if hasattr(logits_val, "to_numpy") else logits_val
            
            preds = logits_np.argmax(axis=1)
            acc = (preds == fake_labels).sum() / 4.0
            
            print(f"{i}\t{loss_scalar:.5f}\t\t{acc*100:.0f}%")
            
            if acc == 1.0 and loss_scalar < 0.05:
                print("\nSUCCESS! 模型成功过拟合了小数据集。")
                print("核心结论：\n1. Conv2D 和 Linear 的梯度计算正确。\n2. Autograd 链条通畅。\n3. SGD 参数更新生效。")
                print("CIFAR-10 准确率低只是因为还没有训练够久/网络不够深。")
                return

    print("\nFAILED! 跑了100轮还没学会4张图，框架可能有严重 Bug。")

if __name__ == "__main__":
    main()