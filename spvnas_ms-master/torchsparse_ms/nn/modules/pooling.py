from torch import nn

from torchsparse_ms import SparseTensor
from torchsparse_ms.nn import functional as F

__all__ = ['GlobalAvgPool', 'GlobalMaxPool']


class GlobalAvgPool(nn.Module):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.global_avg_pool(input)


class GlobalMaxPool(nn.Module):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.global_max_pool(input)
