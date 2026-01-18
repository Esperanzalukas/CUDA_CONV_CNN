import my_deep_lib as cuda
from .module import Module
from core.basic_operator import Op, Value

class MaxPool2DOp(Op):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.mask = None
    
    def compute(self, x):
        # C++ extension 现在的 maxpool_forward 返回 (output, mask)
        out, mask = cuda.maxpool_forward(x, self.kernel_size, self.stride)
        
        # 我们必须只返回 output 给计算图的下一层
        # mask 保存起来用于反向传播
        self.mask = mask 
        return out

    def gradient(self, out_grad, node):
        grad_out = out_grad.realize_cached_data()
        
        # C++ extension 现在的 maxpool_backward 接受 (grad_output, mask, k, s)
        # 注意这里传入的是 self.mask，而不是 self.x
        dx = cuda.maxpool_backward(grad_out, self.mask, self.kernel_size, self.stride)
        
        gx = Value()
        gx._init(None, [], cached_data=dx)
        return (gx,)

class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else kernel_size
    
    def forward(self, x):
        return MaxPool2DOp(self.k, self.s)(x)