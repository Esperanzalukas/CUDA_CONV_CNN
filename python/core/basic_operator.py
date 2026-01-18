from typing import List, Optional
import numpy as np
import my_deep_lib

class Value:
    """基础 Value 类（计算图节点）"""
    def __init__(self):
        self.cached_data = None
        self.requires_grad = True
        self.grad = None
        self.op = None
        self.inputs = []
    
    def _init(self, op, inputs, *, cached_data=None, requires_grad=True):
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad
    
    def realize_cached_data(self):
        return self.cached_data

    # === 新增: 运算符重载 ===
    def __add__(self, other):
        return add(self, other)

    def backward(self, out_grad=None):
        # 1. 初始化梯度
        if out_grad is None:
            # 根据 cached_data 类型创建全1的梯度
            if isinstance(self.cached_data, my_deep_lib.Tensor):
                self.grad = my_deep_lib.Tensor(self.cached_data.shape(), self.cached_data.device())
                self.grad.fill(1.0)
            else:
                self.grad = np.ones_like(self.cached_data, dtype=np.float32)
        else:
            self.grad = out_grad

        # 2. 拓扑排序
        topo_order = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.inputs:
                    build_topo(child)
                topo_order.append(node)
        
        build_topo(self)
        
        # 3. 反向传播
        for node in reversed(topo_order):
            if node.grad is None: continue
            if node.op is None: continue 
            
            grads = node.op.gradient(as_value(node.grad), node)
            
            for i, input_node in enumerate(node.inputs):
                if not input_node.requires_grad: continue
                
                if i >= len(grads) or grads[i] is None:
                    continue

                g = grads[i].realize_cached_data()
                
                # 累加梯度
                if input_node.grad is None:
                    input_node.grad = g
                else:
                    input_node_grad_is_tensor = isinstance(input_node.grad, my_deep_lib.Tensor)
                    g_is_tensor = isinstance(g, my_deep_lib.Tensor)
                    
                    if input_node_grad_is_tensor and g_is_tensor:
                        # 使用 C++ elementwise_add
                        input_node.grad = my_deep_lib.elementwise_add(input_node.grad, g)
                    elif not input_node_grad_is_tensor and not g_is_tensor:
                        input_node.grad = input_node.grad + g
                    else:
                        # 混合类型，强制转 Numpy 计算 (Fallback)
                        v1 = input_node.grad.to_numpy() if input_node_grad_is_tensor else input_node.grad
                        v2 = g.to_numpy() if g_is_tensor else g
                        input_node.grad = v1 + v2

def as_value(data):
    v = Value()
    v.cached_data = data
    return v

class Op:
    def __call__(self, *args):
        inputs = [arg for arg in args]
        raw_inputs = [x.realize_cached_data() for x in inputs]
        output_data = self.compute(*raw_inputs)
        requires_grad = any(x.requires_grad for x in inputs)
        output_val = Value()
        output_val._init(self, inputs, cached_data=output_data, requires_grad=requires_grad)
        return output_val
    
    def compute(self, *args):
        raise NotImplementedError
    
    def gradient(self, out_grad, node):
        raise NotImplementedError

# === 新增: ElemwiseAdd ===
class ElemwiseAdd(Op):
    def compute(self, a, b):
        # 混合类型检查
        a_is_tensor = isinstance(a, my_deep_lib.Tensor)
        b_is_tensor = isinstance(b, my_deep_lib.Tensor)
        
        if a_is_tensor and b_is_tensor:
            return my_deep_lib.elementwise_add(a, b)
        elif not a_is_tensor and not b_is_tensor:
            return a + b
        else:
            # Fallback to numpy if types differ
            av = a.to_numpy() if a_is_tensor else a
            bv = b.to_numpy() if b_is_tensor else b
            return av + bv

    def gradient(self, out_grad, node):
        return out_grad, out_grad

def add(a, b):
    return ElemwiseAdd()(a, b)