import math

from torch.nn import Parameter

def glorot(tensor: Parameter) -> None:
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor: Parameter) -> None:
    if tensor is not None:
        tensor.data.fill_(0)