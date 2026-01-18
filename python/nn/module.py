import numpy as np
from basic_operator import Value
import my_deep_lib  # 引入库以进行类型检查

class Parameter(Value):
    def __init__(self, data):
        super().__init__()
        
        # 如果传入的已经是 Tensor 对象 (例如 C++ GPU Tensor)，直接使用
        if isinstance(data, my_deep_lib.Tensor):
            self.cached_data = data
        else:
            # 否则假设是 Numpy 兼容的数据，转换为 float32 数组
            self.cached_data = np.array(data, dtype=np.float32)
            
        self.requires_grad = True

class Module:
    def __init__(self):
        self._modules = {}
        self._params = {}
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
    
    def parameters(self):
        params = []
        for p in self._params.values():
            params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params
    
    def train(self): 
        for m in self._modules.values():
            m.train()
            
    def eval(self): 
        for m in self._modules.values():
            m.eval()
            
    def __call__(self, *args):
        return self.forward(*args)
        
    def forward(self, *args):
        raise NotImplementedError