#!/usr/bin/env python
"""
VGG Runner
Usage: python run_vgg.py cifar|imagenet [epochs]
"""
import sys
import os
import numpy as np
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def train_vgg_cifar(epochs=30, batch_size=64, lr=0.01):
    """Train VGG on CIFAR-10"""
    # 直接运行现有脚本
    os.system("python train/train_vgg_cifar10.py")

def train_vgg_imagenet(epochs=30, batch_size=32, lr=0.01):
    """Train VGG on Tiny-ImageNet"""
    import json
    import logging
    import my_deep_lib
    from core.basic_operator import Value
    from nn import Module, Parameter, Conv2D, Linear, ReLU, MaxPool2D, cross_entropy_loss
    from optim import SGD
    from data import DataLoader
    from data.imagenet import TinyImageNetDataset, download_tiny_imagenet
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        USE_TB = True
    except:
        USE_TB = False
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/VGG_TinyImageNet_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('vgg_imagenet')
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    writer = SummaryWriter(log_dir=log_dir) if USE_TB else None
    
    print("="*60)
    print("VGG Training on Tiny-ImageNet")
    print(f"Log directory: {log_dir}")
    print("="*60)
    
    # Data
    print("\n[1] Loading data...")
    data_dir = './data/tiny-imagenet-200'
    if not os.path.exists(data_dir):
        download_tiny_imagenet('./data')
    
    train_ds = TinyImageNetDataset(data_dir, train=True)
    test_ds = TinyImageNetDataset(data_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Flatten
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
    
    # VGG Model for 64x64 input
    class VGG(Module):
        def __init__(self, num_classes=200):
            super().__init__()
            # Block 1: 64x64 -> 32x32
            self.conv1_1 = Conv2D(3, 32, 3, padding=1, use_bias=False)
            self.conv1_2 = Conv2D(32, 32, 3, padding=1, use_bias=False)
            self.pool1 = MaxPool2D(2, 2)
            
            # Block 2: 32x32 -> 16x16
            self.conv2_1 = Conv2D(32, 64, 3, padding=1, use_bias=False)
            self.conv2_2 = Conv2D(64, 64, 3, padding=1, use_bias=False)
            self.pool2 = MaxPool2D(2, 2)
            
            # Block 3: 16x16 -> 8x8
            self.conv3_1 = Conv2D(64, 128, 3, padding=1, use_bias=False)
            self.conv3_2 = Conv2D(128, 128, 3, padding=1, use_bias=False)
            self.pool3 = MaxPool2D(2, 2)
            
            # Block 4: 8x8 -> 4x4
            self.conv4_1 = Conv2D(128, 256, 3, padding=1, use_bias=False)
            self.conv4_2 = Conv2D(256, 256, 3, padding=1, use_bias=False)
            self.pool4 = MaxPool2D(2, 2)
            
            # FC: 256*4*4 = 4096
            self.flatten = Flatten()
            self.fc1 = Linear(256 * 4 * 4, 512, use_bias=False)
            self.fc2 = Linear(512, num_classes, use_bias=False)
            self.relu = ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv1_1(x))
            x = self.pool1(self.relu(self.conv1_2(x)))
            x = self.relu(self.conv2_1(x))
            x = self.pool2(self.relu(self.conv2_2(x)))
            x = self.relu(self.conv3_1(x))
            x = self.pool3(self.relu(self.conv3_2(x)))
            x = self.relu(self.conv4_1(x))
            x = self.pool4(self.relu(self.conv4_2(x)))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    print("\n[2] Building model...")
    model = VGG(num_classes=200)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    num_params = sum(p.data.size for p in model.parameters())
    print(f"  Model: VGG")
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
    
    # Save
    with open(os.path.join(log_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if writer:
        writer.close()
    
    print(f"\nLogs saved to: {log_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_vgg.py <dataset> [epochs]")
        print("  dataset: cifar, imagenet (tiny-imagenet)")
        print("  epochs: default 30")
        sys.exit(1)
    
    dataset = sys.argv[1].lower()
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    if dataset in ['cifar', 'cifar10']:
        train_vgg_cifar(epochs)
    elif dataset in ['imagenet', 'tinyimagenet', 'tiny-imagenet']:
        train_vgg_imagenet(epochs)
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

if __name__ == "__main__":
    main()
