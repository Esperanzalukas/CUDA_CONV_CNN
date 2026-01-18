# CUDA CNN - Custom Deep Learning Framework

A custom deep learning framework with CUDA-accelerated convolution operations.

## Features
- Custom CUDA kernels for Conv2D, MaxPool2D, Linear layers
- BatchNorm, ReLU, Cross-entropy loss
- Models: Simple CNN, VGG, ResNet-18
- Datasets: CIFAR-10, Tiny-ImageNet
- TensorBoard logging

## Quick Start

### 1. Setup Environment (Conda)
```bash
conda env create -f environment.yml
conda activate cuda_conv_cnn
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 2. Build CUDA Library
```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..
```

### 3. Train Models

```bash
# Simple CNN on CIFAR-10
python run_cnn.py cifar

# VGG on CIFAR-10
python run_vgg.py cifar

# VGG on Tiny-ImageNet (auto-downloads ~240MB)
python run_vgg.py imagenet

# ResNet-18 on CIFAR-10
python run_resnet.py cifar

# ResNet-18 on Tiny-ImageNet
python run_resnet.py imagenet
```

### 4. View Training Logs
```bash
tensorboard --logdir=./logs
```

## Project Structure
```
├── cuda_src/          # CUDA kernels
├── python/
│   ├── core/          # Autograd engine
│   ├── nn/            # Layers (Conv2D, Linear, BatchNorm, etc.)
│   ├── optim/         # Optimizers (SGD)
│   └── data/          # DataLoaders
├── train/             # Training scripts
├── run_*.py           # Entry scripts
└── build/             # Compiled library (generated)
```

## Requirements
- CUDA 12.x
- CMake >= 3.18
- Python 3.10+
- See `requirements.txt` for Python packages
