import functools
from torch.nn import functional as F
import mindspore as ms
from mindspore import ops, nn


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper
# Reference: https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/analysis_and_preparation.html

# class MindSpore_Loss(nn.Cell):
#     def __init__(self, weight=None, gamma=2.0, alpha=0.25, reduction='mean'):
#         super(MindSpore_Loss, self).__init__()
#         self.sigmoid = ops.Sigmoid()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = ms.Tensor(weight) if weight is not None else weight
#         self.reduction = reduction
#         self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(reduction="none")
#         self.is_weight = (weight is not None)

#     def reduce_loss(self, loss):
#         """Reduce loss as specified.
#         Args:
#             loss (Tensor): Elementwise loss tensor.
#         Return:
#             Tensor: Reduced loss tensor.
#         """
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss

#     def weight_reduce_loss(self, loss):
    
#         loss = self.reduce_loss(loss)
        
#         return loss

#     def construct(self, pred, target):
#         pred_sigmoid = self.sigmoid(pred)
#         target = ops.cast(target, pred.dtype)
#         pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
#         focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ops.pow(pt, self.gamma)
#         loss = self.binary_cross_entropy_with_logits(pred, target) * focal_weight
#         if self.is_weight:
#             weight = self.weight
#             if self.weight.shape != loss.shape:
#                 if self.weight.shape[0] == loss.shape[0]:
#                     # For most cases, weight is of shape (num_priors, ),
#                     #  which means it does not have the second axis num_class
#                     weight = self.weight.view(-1, 1)
#                 elif self.weight.size == loss.size:
#                     # Sometimes, weight per anchor per class is also needed. e.g.
#                     #  in FSAF. But it may be flattened of shape
#                     #  (num_priors x num_class, ), while loss is still of shape
#                     #  (num_priors, num_class).
#                     weight = self.weight.view(loss.shape[0], -1)
#                 elif self.weight.ndim != loss.ndim:
#                     raise ValueError(f"weight shape {self.weight.shape} is not match to loss shape {loss.shape}")
#             loss = loss * weight
#         loss = self.weight_reduce_loss(loss)
#         return loss