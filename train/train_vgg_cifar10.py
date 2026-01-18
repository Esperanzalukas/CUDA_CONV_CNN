"""
VGG-style CNN Training on CIFAR-10
支持 wandb 可视化训练过程
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import my_deep_lib 
from core.basic_operator import Value
from nn import Module, Parameter, Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss
from optim import SGD
from data import DataLoader, CIFAR10Dataset

# ============================================================================
# TensorBoard + 日志配置
# ============================================================================
import json
import logging
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    print("Warning: tensorboard not installed. Run `pip install tensorboard` to enable.")
    USE_TENSORBOARD = False

# 全局变量
writer = None
logger = None

def setup_logging(log_dir):
    """设置日志系统"""
    global logger
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 文件 handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def init_tensorboard(log_dir):
    """初始化 TensorBoard"""
    global writer
    if USE_TENSORBOARD:
        writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_metrics(metrics, step=None):
    """记录指标到 TensorBoard"""
    global writer
    if writer is not None:
        for key, value in metrics.items():
            if '/' in key:
                writer.add_scalar(key, value, step)
            else:
                writer.add_scalar(f'metrics/{key}', value, step)

# ============================================================================
# Flatten 操作
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
            raise RuntimeError(f"Unknown shape type for data: {type(d)}")
            
        if hasattr(d, "numel"):
             numel = d.numel()
        else:
             numel = d.size
        
        current_batch = self.shape[0]
        flatten_dim = numel // current_batch
        
        if isinstance(d, my_deep_lib.Tensor):
            if numel % current_batch != 0:
                 raise RuntimeError(f"Flatten error: numel {numel} not divisible by batch {current_batch}")
            new_d = d.reshape([current_batch, flatten_dim]) 
        else:
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

# ============================================================================
# VGG-style 模型定义
# ============================================================================
class VGGNet(Module):
    """
    VGG风格的深度CNN模型
    结构: [Conv-ReLU-Conv-ReLU-Pool] x 3 -> FC -> FC -> Output
    特点: 使用小卷积核(3x3)堆叠，增加网络深度
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Block 1: 3 -> 64 -> 64, 32x32 -> 16x16
        self.conv1_1 = Conv2D(3, 64, 3, padding=1, bias=False)
        self.relu1_1 = ReLU()
        self.conv1_2 = Conv2D(64, 64, 3, padding=1, bias=False)
        self.relu1_2 = ReLU()
        self.pool1 = MaxPool2D(2, 2)
        
        # Block 2: 64 -> 128 -> 128, 16x16 -> 8x8
        self.conv2_1 = Conv2D(64, 128, 3, padding=1, bias=False)
        self.relu2_1 = ReLU()
        self.conv2_2 = Conv2D(128, 128, 3, padding=1, bias=False)
        self.relu2_2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)
        
        # Block 3: 128 -> 256 -> 256, 8x8 -> 4x4
        self.conv3_1 = Conv2D(128, 256, 3, padding=1, bias=False)
        self.relu3_1 = ReLU()
        self.conv3_2 = Conv2D(256, 256, 3, padding=1, bias=False)
        self.relu3_2 = ReLU()
        self.pool3 = MaxPool2D(2, 2)
        
        # Classifier: 256 * 4 * 4 = 4096 -> 512 -> num_classes
        self.fc1 = Linear(256 * 4 * 4, 512)
        self.relu_fc1 = ReLU()
        self.fc2 = Linear(512, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.pool3(x)
        
        # Classifier
        x = flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================================
# 训练与评估
# ============================================================================
def evaluate(model, test_loader):
    """评估模型"""
    test_correct, test_total = 0, 0
    for data, labels in test_loader:
        if data.shape[-1] == 3:
            data = data.transpose(0, 3, 1, 2)
        
        gpu_data = my_deep_lib.Tensor.from_numpy(data, my_deep_lib.Device.gpu(0))
        x = Value()
        x._init(None, [], cached_data=gpu_data)
        
        logits = model(x)
        logits_data = logits.realize_cached_data()
        if hasattr(logits_data, "to_numpy"):
            logits_np = logits_data.to_numpy()
        else:
            logits_np = logits_data
            
        pred = logits_np.argmax(axis=1)
        test_correct += (pred == labels).sum()
        test_total += len(labels)
    
    return test_correct / test_total

def train_epoch(model, train_loader, optimizer, epoch):
    """训练一个 epoch"""
    total_loss, correct, total = 0.0, 0, 0
    batch_losses = []
    
    for i, (data, labels) in enumerate(train_loader):
        if data.shape[-1] == 3:
            data = data.transpose(0, 3, 1, 2)

        gpu_data = my_deep_lib.Tensor.from_numpy(data, my_deep_lib.Device.gpu(0))
        gpu_labels = my_deep_lib.Tensor.from_numpy(labels.astype(np.float32), my_deep_lib.Device.gpu(0))

        x = Value()
        x._init(None, [], cached_data=gpu_data)
        y = Value()
        y._init(None, [], cached_data=gpu_labels, requires_grad=False)
        
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 获取 loss 值
        loss_val = loss.realize_cached_data()
        if hasattr(loss_val, "to_numpy"):
            loss_scalar = float(loss_val.to_numpy())
        else:
            loss_scalar = float(loss_val)
        total_loss += loss_scalar
        batch_losses.append(loss_scalar)
        
        # 计算准确率
        logits_data = logits.realize_cached_data()
        if hasattr(logits_data, "to_numpy"):
            logits_np = logits_data.to_numpy()
        else:
            logits_np = logits_data
            
        pred = logits_np.argmax(axis=1)
        correct += (pred == labels).sum()
        total += len(labels)
        
        # 每 50 个 batch 记录一次
        if (i + 1) % 50 == 0:
            batch_loss = total_loss / (i + 1)
            batch_acc = correct / total
            print(f"  Batch {i+1}/{len(train_loader)}, Loss: {batch_loss:.4f}, Acc: {100*batch_acc:.2f}%")
            
            # 记录 batch 级别的指标
            global_step = epoch * len(train_loader) + i
            log_metrics({
                "batch/loss": batch_loss,
                "batch/accuracy": batch_acc,
            }, step=global_step)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # ========== 配置 ==========
    config = {
        "model_name": "VGG6",
        "batch_size": 32,
        "epochs": 20,
        "lr": 0.001,
        "momentum": 0.9,
        "num_classes": 10,
    }
    
    print("=" * 60)
    print("VGG-style CNN Training on CIFAR-10")
    print("=" * 60)
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/{config['model_name']}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化 TensorBoard 和日志
    init_tensorboard(log_dir)
    setup_logging(log_dir)
    
    # 保存配置
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Log directory: {log_dir}")
    
    # ========== 数据加载 ==========
    print("\n[1] Loading data...")
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    train_ds = CIFAR10Dataset('./data', train=True)
    test_ds = CIFAR10Dataset('./data', train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Batch size: {config['batch_size']}")
    
    # ========== 模型初始化 ==========
    print("\n[2] Building model...")
    model = VGGNet(num_classes=config["num_classes"])
    
    # 计算参数量
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
    
    # ========== 训练循环 ==========
    print("\n[3] Training...")
    print("-" * 60)
    
    best_test_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    
    for epoch in range(config["epochs"]):
        t0 = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        
        # 评估
        test_acc = evaluate(model, test_loader)
        
        epoch_time = time.time() - t0
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        # 更新最佳准确率
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # 打印结果
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Loss={train_loss:.4f}, Train Acc={100*train_acc:.2f}%, "
              f"Test Acc={100*test_acc:.2f}%, Time={epoch_time:.1f}s")
        
        # 记录到 TensorBoard
        log_metrics({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "test/accuracy": test_acc,
            "time/epoch": epoch_time,
        }, step=epoch + 1)
        
        # 记录到日志文件
        if logger:
            logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, TrainAcc={100*train_acc:.2f}%, TestAcc={100*test_acc:.2f}%")
    
    print("-" * 60)
    print(f"\nTraining completed!")
    print(f"Best Test Accuracy: {100*best_test_acc:.2f}%")
    
    # 关闭 TensorBoard
    if writer:
        writer.close()
    
    # 保存最终结果
    if logger:
        logger.info(f"Training completed! Best Test Acc: {100*best_test_acc:.2f}%")
    
    # ========== 保存本地训练曲线和数据 ==========
    save_training_curves(history, config, log_dir)
    
    print(f"\nAll logs saved to: {log_dir}")
    print("To view TensorBoard: tensorboard --logdir=./logs")
    
    return history

def save_training_curves(history, config, log_dir):
    """保存训练曲线和数据到日志目录"""
    # 保存 history 数据为 JSON
    history_path = os.path.join(log_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to: {history_path}")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Loss 曲线
        axes[0].plot(epochs, history["train_loss"], 'b-', linewidth=2, label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy 曲线
        axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], 'b-', linewidth=2, label='Train Acc')
        axes[1].plot(epochs, [a * 100 for a in history["test_acc"]], 'r-', linewidth=2, label='Test Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Train vs Test Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片到日志目录
        save_path = os.path.join(log_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150)
        print(f"Curves saved to: {save_path}")
        plt.close()
        
    except ImportError:
        print("Note: matplotlib not installed. Skipping curve plotting.")

if __name__ == "__main__":
    main()
