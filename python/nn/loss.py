import numpy as np
import my_deep_lib as cuda
from basic_operator import Op, Value

class CrossEntropyOp(Op):
    def compute(self, logits, labels):
        self.logits = logits
        
        # 稳健性：确保标签在 GPU 上
        if isinstance(labels, np.ndarray):
            self.labels = cuda.Tensor.from_numpy(labels.astype(np.float32), logits.device())
        else:
            self.labels = labels
            
        # 【重要】C++ 的 cross_entropy_loss 已经除以了 Batch Size (返回的是 Mean)
        # 所以这里直接返回即可，不要再除以 N
        return cuda.cross_entropy_loss(self.logits, self.labels)
    
    def gradient(self, out_grad, node):
        # 【重要】C++ 的 cross_entropy_backward 也已经除以了 Batch Size
        # 返回的就是标准的 Mean Gradient
        d_logits = cuda.cross_entropy_backward(self.logits, self.labels)
        
        gv = Value()
        gv._init(None, [], cached_data=d_logits)
        return (gv, None)

def cross_entropy_loss(logits, labels):
    return CrossEntropyOp()(logits, labels)