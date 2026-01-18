"""
简单的 CNN 模型（用于 CIFAR-10）

架构：
Conv(3->32) -> ReLU -> MaxPool -> Conv(32->64) -> ReLU -> MaxPool -> Flatten -> Linear(64*8*8->512) -> ReLU -> Linear(512->10)
"""
import numpy as np
from nn.module import Module, Parameter
from nn.conv import Conv2D, MaxPool2D
from nn.linear import Linear
from nn.activations import ReLU
from basic_operator import Value

class SimpleCNN(Module):
    """
    简单的卷积神经网络
    
    输入：[batch, 3, 32, 32] (CIFAR-10 图像)
    输出：[batch, 10] (10 个类别的 logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 第一层卷积块：3 -> 32 channels
        self.conv1 = Conv2D(in_channels=3, out_channels=32, 
                           kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # 第二层卷积块：32 -> 64 channels
        self.conv2 = Conv2D(in_channels=32, out_channels=64,
                           kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # 全连接层
        self.fc1 = Linear(in_features=64 * 8 * 8, out_features=512)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=512, out_features=num_classes)
    
    def forward(self, x: Value) -> Value:
        """
        前向传播
        
        Args:
            x: [batch, 3, 32, 32]
        
        Returns:
            logits: [batch, num_classes]
        """
        # 第一个卷积块
        x = self.conv1(x)       # [batch, 32, 32, 32]
        x = self.relu1(x)
        x = self.pool1(x)       # [batch, 32, 16, 16]
        
        # 第二个卷积块
        x = self.conv2(x)       # [batch, 64, 16, 16]
        x = self.relu2(x)
        x = self.pool2(x)       # [batch, 64, 8, 8]
        
        # Flatten
        x_data = x.realize_cached_data()
        batch_size = x_data.shape[0]
        flattened = x_data.reshape(batch_size, -1)  # [batch, 64*8*8]
        x = Value.make_const(flattened, requires_grad=x.requires_grad)
        x._init(x.op, x.inputs, cached_data=flattened, requires_grad=x.requires_grad)
        
        # 全连接层
        x = self.fc1(x)         # [batch, 512]
        x = self.relu3(x)
        x = self.fc2(x)         # [batch, 10]
        
        return x

# 简化版（更小的模型，训练更快）
class TinyCNN(Module):
    """
    更小的 CNN（用于快速测试）
    
    Conv(3->16) -> ReLU -> MaxPool -> Conv(16->32) -> ReLU -> MaxPool -> Linear(32*8*8->10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv2 = Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        
        self.fc = Linear(32 * 8 * 8, num_classes)
    
    def forward(self, x: Value) -> Value:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x_data = x.realize_cached_data()
        batch_size = x_data.shape[0]
        flattened = x_data.reshape(batch_size, -1)
        x = Value.make_const(flattened, requires_grad=x.requires_grad)
        x._init(x.op, x.inputs, cached_data=flattened, requires_grad=x.requires_grad)
        
        x = self.fc(x)
        return x