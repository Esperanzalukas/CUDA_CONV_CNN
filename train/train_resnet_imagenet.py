"""
ResNet Training on Tiny-ImageNet / ImageNet-100
支持 TensorBoard 日志
"""
import sys, os, time
import json
import logging
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import my_deep_lib 
from core.basic_operator import Value, Op
from nn import Module, Parameter, Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss
from nn.batchnorm import BatchNorm2D
from optim import SGD
from data import DataLoader
from data.imagenet import TinyImageNetDataset, download_tiny_imagenet

# ============================================================================
# TensorBoard + 日志
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

# ============================================================================
# 残差加法 (Skip Connection)
# ============================================================================
class AddOp(Op):
    """元素加法，用于残差连接"""
    def compute(self, a, b):
        is_gpu = isinstance(a, my_deep_lib.Tensor)
        if is_gpu:
            return my_deep_lib.elementwise_add(a, b)
        return a + b
    
    def gradient(self, out_grad, node):
        # d(a+b)/da = 1, d(a+b)/db = 1
        return (out_grad, out_grad)

def residual_add(x, residual):
    """残差加法: x + residual"""
    op = AddOp()
    return op(x, residual)

# ============================================================================
# Flatten
# ============================================================================
class Flatten:
    def __call__(self, x):
        d = x.realize_cached_data()
        shape_attr = getattr(d, "shape", None)
        if isinstance(shape_attr, (list, tuple)):
            self.shape = shape_attr
        elif callable(shape_attr):
            self.shape = shape_attr()
        else:
            raise RuntimeError(f"Unknown shape type: {type(d)}")
        
        numel = d.numel() if hasattr(d, "numel") else d.size
        batch = self.shape[0]
        flatten_dim = numel // batch
        
        if isinstance(d, my_deep_lib.Tensor):
            new_d = d.reshape([batch, flatten_dim])
        else:
            new_d = d.reshape(batch, -1).astype(np.float32)
        
        r = Value()
        r._init(None, [x], cached_data=new_d)
        r.op = self
        return r
    
    def gradient(self, out_grad, node):
        g = out_grad.realize_cached_data()
        if isinstance(g, my_deep_lib.Tensor):
            dg = g.reshape(list(self.shape))
        else:
            dg = g.reshape(self.shape).astype(np.float32)
        gv = Value()
        gv._init(None, [], cached_data=dg)
        return (gv,)

flatten = Flatten()

# ============================================================================
# Global Average Pooling
# ============================================================================
class GlobalAvgPool2DOp(Op):
    """全局平均池化: [N,C,H,W] -> [N,C,1,1]"""
    def compute(self, x):
        is_gpu = isinstance(x, my_deep_lib.Tensor)
        if is_gpu:
            x_np = x.to_numpy()
            device = x.device()
        else:
            x_np = x
            device = None
        
        self.input_shape = x_np.shape
        N, C, H, W = x_np.shape
        
        # 对 H, W 取平均
        out = np.mean(x_np, axis=(2, 3), keepdims=True)
        
        if is_gpu:
            return my_deep_lib.Tensor.from_numpy(out.astype(np.float32), device)
        return out
    
    def gradient(self, out_grad, node):
        g = out_grad.realize_cached_data()
        is_gpu = isinstance(g, my_deep_lib.Tensor)
        if is_gpu:
            g_np = g.to_numpy()
            device = g.device()
        else:
            g_np = g
            device = None
        
        N, C, H, W = self.input_shape
        # 梯度均分到每个空间位置
        dx = np.broadcast_to(g_np / (H * W), self.input_shape).copy()
        
        if is_gpu:
            dx_tensor = my_deep_lib.Tensor.from_numpy(dx.astype(np.float32), device)
            gv = Value()
            gv._init(None, [], cached_data=dx_tensor)
            return (gv,)
        
        gv = Value()
        gv._init(None, [], cached_data=dx)
        return (gv,)

class GlobalAvgPool2D(Module):
    def forward(self, x):
        op = GlobalAvgPool2DOp()
        return op(x)

# ============================================================================
# ResNet Basic Block
# ============================================================================
class BasicBlock(Module):
    """
    ResNet BasicBlock (用于 ResNet-18, 34)
    结构: Conv-BN-ReLU-Conv-BN + shortcut -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2D(out_channels)
        self.downsample = downsample  # 用于匹配维度
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = residual_add(out, identity)
        out = self.relu(out)
        
        return out

class Downsample(Module):
    """下采样模块: 1x1 Conv + BN"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = BatchNorm2D(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))

# ============================================================================
# ResNet-18
# ============================================================================
class ResNet18(Module):
    """
    ResNet-18 for Tiny-ImageNet (64x64) or ImageNet (224x224)
    """
    def __init__(self, num_classes=200, input_size=64):
        super().__init__()
        self.input_size = input_size
        
        # 根据输入大小调整初始卷积
        if input_size == 64:  # Tiny-ImageNet
            self.conv1 = Conv2D(3, 64, 3, stride=1, padding=1, bias=False)
            self.pool1 = None
        else:  # ImageNet 224x224
            self.conv1 = Conv2D(3, 64, 7, stride=2, padding=3, bias=False)
            self.pool1 = MaxPool2D(3, 2)  # 简化版
        
        self.bn1 = BatchNorm2D(64)
        self.relu = ReLU()
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = GlobalAvgPool2D()
        self.fc = Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = Downsample(in_channels, out_channels, stride)
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return layers  # 返回 list，在 forward 中手动遍历
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.pool1:
            x = self.pool1(x)
        
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        
        x = self.avgpool(x)
        x = flatten(x)
        x = self.fc(x)
        return x

# ============================================================================
# 训练函数
# ============================================================================
def train_epoch(model, train_loader, optimizer, epoch):
    total_loss, correct, total = 0.0, 0, 0
    
    for i, (data, labels) in enumerate(train_loader):
        # 确保数据是 CHW 格式
        if len(data.shape) == 4 and data.shape[-1] == 3:
            data = data.transpose(0, 3, 1, 2)
        
        gpu_data = my_deep_lib.Tensor.from_numpy(data.astype(np.float32), my_deep_lib.Device.gpu(0))
        gpu_labels = my_deep_lib.Tensor.from_numpy(labels.astype(np.float32), my_deep_lib.Device.gpu(0))
        
        x = Value(); x._init(None, [], cached_data=gpu_data)
        y = Value(); y._init(None, [], cached_data=gpu_labels, requires_grad=False)
        
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.realize_cached_data()
        loss_scalar = float(loss_val.to_numpy()) if hasattr(loss_val, "to_numpy") else float(loss_val)
        total_loss += loss_scalar
        
        logits_data = logits.realize_cached_data()
        logits_np = logits_data.to_numpy() if hasattr(logits_data, "to_numpy") else logits_data
        
        pred = logits_np.argmax(axis=1)
        correct += (pred == labels).sum()
        total += len(labels)
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}, Loss: {total_loss/(i+1):.4f}, Acc: {100*correct/total:.2f}%")
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader):
    correct, total = 0, 0
    for data, labels in test_loader:
        if len(data.shape) == 4 and data.shape[-1] == 3:
            data = data.transpose(0, 3, 1, 2)
        
        gpu_data = my_deep_lib.Tensor.from_numpy(data.astype(np.float32), my_deep_lib.Device.gpu(0))
        x = Value(); x._init(None, [], cached_data=gpu_data)
        
        logits = model(x)
        logits_data = logits.realize_cached_data()
        logits_np = logits_data.to_numpy() if hasattr(logits_data, "to_numpy") else logits_data
        
        pred = logits_np.argmax(axis=1)
        correct += (pred == labels).sum()
        total += len(labels)
    
    return correct / total

# ============================================================================
# Main
# ============================================================================
def main():
    config = {
        "model_name": "ResNet18",
        "dataset": "TinyImageNet",
        "batch_size": 32,
        "epochs": 30,
        "lr": 0.01,
        "momentum": 0.9,
        "num_classes": 200,
        "input_size": 64,
    }
    
    print("=" * 60)
    print("ResNet-18 Training on Tiny-ImageNet")
    print("=" * 60)
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/{config['model_name']}_{config['dataset']}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    init_tensorboard(log_dir)
    setup_logging(log_dir)
    
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Log directory: {log_dir}")
    
    # ========== 下载并加载数据 ==========
    print("\n[1] Loading data...")
    data_dir = download_tiny_imagenet('./data')
    
    train_ds = TinyImageNetDataset(data_dir, train=True)
    test_ds = TinyImageNetDataset(data_dir, train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    # ========== 模型初始化 ==========
    print("\n[2] Building model...")
    model = ResNet18(num_classes=config["num_classes"], input_size=config["input_size"])
    
    total_params = 0
    for p in model.parameters():
        d = p.cached_data
        if hasattr(d, "numel"):
            total_params += d.numel()
        else:
            total_params += d.size
    print(f"  Model: {config['model_name']}")
    print(f"  Parameters: {total_params:,}")
    
    if logger:
        logger.info(f"Model: {config['model_name']}, Params: {total_params:,}")
    
    # ========== 优化器 ==========
    optimizer = SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    
    # ========== 训练 ==========
    print("\n[3] Training...")
    print("-" * 60)
    
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc = 0.0
    
    for epoch in range(config["epochs"]):
        t0 = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        test_acc = evaluate(model, test_loader)
        
        epoch_time = time.time() - t0
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        print(f"Epoch {epoch+1}/{config['epochs']}: Loss={train_loss:.4f}, "
              f"Train={100*train_acc:.2f}%, Test={100*test_acc:.2f}%, Time={epoch_time:.1f}s")
        
        log_scalar("train/loss", train_loss, epoch + 1)
        log_scalar("train/accuracy", train_acc, epoch + 1)
        log_scalar("test/accuracy", test_acc, epoch + 1)
        
        if logger:
            logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train={100*train_acc:.2f}%, Test={100*test_acc:.2f}%")
    
    # ========== 保存结果 ==========
    print(f"\nBest Test Accuracy: {100*best_test_acc:.2f}%")
    
    with open(os.path.join(log_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if logger:
        logger.info(f"Training completed! Best Test Acc: {100*best_test_acc:.2f}%")
    
    if writer:
        writer.close()
    
    # 保存曲线
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

if __name__ == "__main__":
    main()
