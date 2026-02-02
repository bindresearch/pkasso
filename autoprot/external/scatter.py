from torch import Tensor, zeros
from typing import Optional

def broadcast(src: Tensor, other: Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
    return scatter_sum(src, index, dim, out, dim_size)