import sys, os, time
import json
import logging
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# 引入 C++ 扩展库
import my_deep_lib 

from core.basic_operator import Value
from nn import Module, Parameter, Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss
from optim import SGD
from data import DataLoader, CIFAR10Dataset

# ============================================================================
# TensorBoard + 日志配置
# ============================================================================
try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    USE_TENSORBOARD = False

writer = None
logger = None

def setup_logging(log_dir):
    global logger
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    return logger

def init_tensorboard(log_dir):
    global writer
    if USE_TENSORBOARD:
        writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_scalar(tag, value, step):
    if writer:
        writer.add_scalar(tag, value, step)

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
        
        # DEBUG (已注释):
        # if not hasattr(self, "printed"):
        #     print(f"DEBUG: Flatten shape: {self.shape} -> [Batch={current_batch}, Dim={flatten_dim}]")
        #     self.printed = True
        
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
        self.conv1 = Conv2D(3, 32, 3, padding=1, bias=False)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)
        
        # Conv2: 32 -> 64
        self.conv2 = Conv2D(32, 64, 3, padding=1, bias=False)
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
    config = {"model_name": "SimpleCNN", "batch_size": 64, "epochs": 20, "lr": 0.001}
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/{config['model_name']}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    init_tensorboard(log_dir)
    setup_logging(log_dir)
    
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Log directory: {log_dir}")
    print("Loading data...")
    
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    train_ds = CIFAR10Dataset('./data', train=True)
    test_ds = CIFAR10Dataset('./data', train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    
    model = CNN()
    
    total_params = 0
    for p in model.parameters():
        d = p.cached_data
        if hasattr(d, "numel"):
            total_params += d.numel()
        else:
            total_params += d.size
    print(f"Parameters: {total_params:,}")
    if logger:
        logger.info(f"Model: {config['model_name']}, Params: {total_params:,}")
    
    opt = SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    
    print("Training...")
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc = 0.0
    
    for epoch in range(config["epochs"]):
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        
        for i, (data, labels) in enumerate(train_loader):
            if data.shape[-1] == 3:
                data = data.transpose(0, 3, 1, 2)

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
            loss_scalar = float(loss_val.to_numpy()) if hasattr(loss_val, "to_numpy") else float(loss_val)
            total_loss += loss_scalar
            
            logits_data = logits.realize_cached_data()
            logits_np = logits_data.to_numpy() if hasattr(logits_data, "to_numpy") else logits_data
                
            pred = logits_np.argmax(axis=1)
            correct += (pred == labels).sum()
            total += len(labels)
            
            if (i+1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {total_loss/(i+1):.4f}, Acc: {100*correct/total:.2f}%")
        
        # 评估
        test_correct, test_total = 0, 0
        for data, labels in test_loader:
            if data.shape[-1] == 3:
                data = data.transpose(0, 3, 1, 2)
            gpu_data = my_deep_lib.Tensor.from_numpy(data, my_deep_lib.Device.gpu(0))
            x = Value(); x._init(None, [], cached_data=gpu_data)
            
            logits = model(x)
            logits_data = logits.realize_cached_data()
            logits_np = logits_data.to_numpy() if hasattr(logits_data, "to_numpy") else logits_data
                
            pred = logits_np.argmax(axis=1)
            test_correct += (pred == labels).sum()
            test_total += len(labels)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        test_acc = test_correct / test_total
        epoch_time = time.time() - t0
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
              f"Train Acc={100*train_acc:.2f}%, Test Acc={100*test_acc:.2f}%, Time={epoch_time:.1f}s")
        
        # TensorBoard
        log_scalar("train/loss", train_loss, epoch + 1)
        log_scalar("train/accuracy", train_acc, epoch + 1)
        log_scalar("test/accuracy", test_acc, epoch + 1)
        
        if logger:
            logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, TrainAcc={100*train_acc:.2f}%, TestAcc={100*test_acc:.2f}%")
    
    # 保存结果
    print(f"\nBest Test Accuracy: {100*best_test_acc:.2f}%")
    
    with open(os.path.join(log_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if logger:
        logger.info(f"Training completed! Best Test Acc: {100*best_test_acc:.2f}%")
    
    if writer:
        writer.close()
    
    # 保存曲线图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)
        
        axes[0].plot(epochs, history["train_loss"], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, [a*100 for a in history["train_acc"]], 'b-', linewidth=2, label='Train')
        axes[1].plot(epochs, [a*100 for a in history["test_acc"]], 'r-', linewidth=2, label='Test')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150)
        plt.close()
        print(f"Curves saved to: {log_dir}/training_curves.png")
    except ImportError:
        pass
    
    print(f"\nAll logs saved to: {log_dir}")
    print("To view TensorBoard: tensorboard --logdir=./logs")

if __name__ == "__main__":
    main()