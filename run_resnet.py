#!/usr/bin/env python
"""
ResNet Runner
Usage: python run_resnet.py cifar|imagenet [epochs]
"""
import sys
import os
import numpy as np
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def train_resnet_cifar(epochs=50, batch_size=64, lr=0.01):
    """Train ResNet on CIFAR-10"""
    import json
    import logging
    import my_deep_lib
    from core.basic_operator import Value, Op
    from nn import Module, Parameter, Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss
    from nn.batchnorm import BatchNorm2D
    from optim import SGD
    from data import DataLoader, CIFAR10Dataset
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        USE_TB = True
    except:
        USE_TB = False
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ResNet18_CIFAR10_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('resnet_cifar')
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    writer = SummaryWriter(log_dir=log_dir) if USE_TB else None
    
    print("="*60)
    print("ResNet-18 Training on CIFAR-10")
    print(f"Log directory: {log_dir}")
    print("="*60)
    
    # Data
    print("\n[1] Loading data...")
    train_ds = CIFAR10Dataset('./data', train=True)
    test_ds = CIFAR10Dataset('./data', train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    # Ops
    class AddOp(Op):
        def compute(self, a, b):
            if isinstance(a, my_deep_lib.Tensor):
                return my_deep_lib.elementwise_add(a, b)
            return a + b
        def gradient(self, out_grad, node):
            return (out_grad, out_grad)
    
    def residual_add(x, residual):
        return AddOp()(x, residual)
    
    class Flatten:
        def __call__(self, x):
            d = x.realize_cached_data()
            shape = d.shape() if callable(getattr(d, 'shape', None)) else d.shape
            self.shape = shape
            batch = shape[0]
            numel = d.numel() if hasattr(d, 'numel') else d.size
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
            d = out_grad.realize_cached_data()
            if isinstance(d, my_deep_lib.Tensor):
                new_d = d.reshape(list(self.shape))
            else:
                new_d = d.reshape(self.shape).astype(np.float32)
            r = Value()
            r._init(None, [out_grad], cached_data=new_d)
            return (r,)
    
    class GlobalAvgPool2D:
        def __call__(self, x):
            d = x.realize_cached_data()
            shape = d.shape() if callable(getattr(d, 'shape', None)) else d.shape
            self.shape = shape
            if isinstance(d, my_deep_lib.Tensor):
                arr = d.to_numpy()
            else:
                arr = d
            out = arr.mean(axis=(2, 3)).astype(np.float32)
            r = Value()
            r._init(None, [x], cached_data=my_deep_lib.Tensor(out) if isinstance(d, my_deep_lib.Tensor) else out)
            r.op = self
            return r
        def gradient(self, out_grad, node):
            d = out_grad.realize_cached_data()
            if isinstance(d, my_deep_lib.Tensor):
                arr = d.to_numpy()
            else:
                arr = d
            H, W = self.shape[2], self.shape[3]
            grad = np.broadcast_to(arr[:, :, None, None], self.shape) / (H * W)
            grad = grad.astype(np.float32)
            r = Value()
            r._init(None, [out_grad], cached_data=my_deep_lib.Tensor(grad) if isinstance(d, my_deep_lib.Tensor) else grad)
            return (r,)
    
    # BasicBlock
    class BasicBlock(Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = Conv2D(in_ch, out_ch, 3, stride=stride, padding=1, use_bias=False)
            self.bn1 = BatchNorm2D(out_ch)
            self.conv2 = Conv2D(out_ch, out_ch, 3, stride=1, padding=1, use_bias=False)
            self.bn2 = BatchNorm2D(out_ch)
            self.relu = ReLU()
            
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = Conv2D(in_ch, out_ch, 1, stride=stride, padding=0, use_bias=False)
                self.downsample_bn = BatchNorm2D(out_ch)
        
        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample:
                identity = self.downsample_bn(self.downsample(x))
            out = residual_add(out, identity)
            out = self.relu(out)
            return out
    
    # ResNet for CIFAR (32x32)
    class ResNet(Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # 32x32 -> 32x32
            self.conv1 = Conv2D(3, 16, 3, stride=1, padding=1, use_bias=False)
            self.bn1 = BatchNorm2D(16)
            self.relu = ReLU()
            
            # 32x32 -> 32x32
            self.layer1 = self._make_layer(16, 16, 2, stride=1)
            # 32x32 -> 16x16
            self.layer2 = self._make_layer(16, 32, 2, stride=2)
            # 16x16 -> 8x8
            self.layer3 = self._make_layer(32, 64, 2, stride=2)
            
            self.gap = GlobalAvgPool2D()
            self.fc = Linear(64, num_classes, use_bias=False)
        
        def _make_layer(self, in_ch, out_ch, num_blocks, stride):
            blocks = [BasicBlock(in_ch, out_ch, stride)]
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(out_ch, out_ch, 1))
            return blocks
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            x = self.gap(x)
            x = self.fc(x)
            return x
    
    print("\n[2] Building model...")
    model = ResNet(num_classes=10)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    num_params = sum(p.data.size for p in model.parameters())
    print(f"  Model: ResNet-18 (CIFAR variant)")
    print(f"  Parameters: {num_params:,}")
    
    # Training
    print("\n[3] Training...")
    print("-"*60)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        t0 = datetime.now()
        
        for i, (data, labels) in enumerate(train_loader):
            x = Value(my_deep_lib.Tensor(data.astype(np.float32)))
            out = model(x)
            loss = cross_entropy_loss(out, labels.astype(np.int32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss.realize_cached_data().to_numpy().mean())
            train_loss += loss_val
            
            pred = out.realize_cached_data().to_numpy()
            correct += (pred.argmax(axis=1) == labels).sum()
            total += len(labels)
            
            if (i + 1) % 50 == 0:
                print(f"\r  Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss_val:.4f}", end='')
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Evaluate
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        for data, labels in test_loader:
            x = Value(my_deep_lib.Tensor(data.astype(np.float32)))
            out = model(x)
            loss = cross_entropy_loss(out, labels.astype(np.int32))
            test_loss += float(loss.realize_cached_data().to_numpy().mean())
            pred = out.realize_cached_data().to_numpy()
            correct += (pred.argmax(axis=1) == labels).sum()
            total += len(labels)
        
        test_loss /= len(test_loader)
        test_acc = correct / total
        
        elapsed = (datetime.now() - t0).seconds
        print(f"\r  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | Time: {elapsed}s")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    print("-"*60)
    print(f"Best Test Accuracy: {best_acc:.4f}")
    
    with open(os.path.join(log_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if writer:
        writer.close()
    
    print(f"\nLogs saved to: {log_dir}")

def train_resnet_imagenet(epochs=30, batch_size=32, lr=0.01):
    """Train ResNet on Tiny-ImageNet"""
    os.system(f"python train/train_resnet_imagenet.py")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_resnet.py <dataset> [epochs]")
        print("  dataset: cifar, imagenet (tiny-imagenet)")
        print("  epochs: default 50 for cifar, 30 for imagenet")
        sys.exit(1)
    
    dataset = sys.argv[1].lower()
    
    if dataset in ['cifar', 'cifar10']:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        train_resnet_cifar(epochs)
    elif dataset in ['imagenet', 'tinyimagenet', 'tiny-imagenet']:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        train_resnet_imagenet(epochs)
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

if __name__ == "__main__":
    main()
