#!/usr/bin/env python
"""
Simple CNN Runner
Usage: python run_cnn.py cifar|imagenet [epochs]
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_cnn.py <dataset> [epochs]")
        print("  dataset: cifar, imagenet (tiny-imagenet)")
        print("  epochs: number of epochs (default: 10 for cifar, 30 for imagenet)")
        sys.exit(1)
    
    dataset = sys.argv[1].lower()
    
    if dataset in ['cifar', 'cifar10']:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"\n{'='*60}")
        print(f"Training Simple CNN on CIFAR-10 for {epochs} epochs")
        print(f"{'='*60}\n")
        os.system(f"python train/train_cifar10.py")
        
    elif dataset in ['imagenet', 'tinyimagenet', 'tiny-imagenet']:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        print(f"\n{'='*60}")
        print(f"Simple CNN is not recommended for ImageNet.")
        print(f"Please use VGG or ResNet: python run_vgg.py imagenet")
        print(f"                         python run_resnet.py imagenet")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"Unknown dataset: {dataset}")
        print("Available: cifar, imagenet")
        sys.exit(1)

if __name__ == "__main__":
    main()
