# import torch
import mindspore as ms
from mindspore import ops as ops
def _pack_bayer(raw):
    h, w = raw.size(-2), raw.size(-1)
    out = ops.cat((raw[..., None, 0:h:2, 0:w:2], # R
                     raw[..., None, 0:h:2, 1:w:2], # G1
                     raw[..., None, 1:h:2, 1:w:2], # B
                     raw[..., None, 1:h:2, 0:w:2]  # G2
                    # ), dim=-3)
                    ), axis=-3)

    return out

def shot_noise(x, k):
    return ops.random_poisson(ms.Tensor([]), x / k) * k - x

def gaussian_noise(x, scale, loc=0):
    # return torch.randn_like(x) * scale + loc
    return ops.randn_like(x) * scale + loc

def tukey_lambda_noise(x, scale, t_lambda=1.4):
    def tukey_lambda_ppf(p, t_lambda):
        # assert not torch.any(t_lambda == 0.0)
        assert not ops.any(t_lambda == 0.0)

        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)

    epsilon = 1e-10
    # U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    U = ops.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = tukey_lambda_ppf(U, t_lambda) * scale

    return Y

def quant_noise(x, q):
    # return (torch.rand_like(x) - 0.5) * q
    return (ops.rand_like(x) - 0.5) * q


def row_noise(x, scale, loc=0):
    if x.dim() == 4:
        B, _, H, W = x.shape
        # noise = (torch.randn((B, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, W * 2))
        noise = (ops.randn((B, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, W * 2))
        return _pack_bayer(noise)
    elif x.dim() == 5:
        B, T, _, H, W = x.shape
        # noise = (torch.randn((B, T, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, 1, W * 2))
        noise = (ops.randn((B, T, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, 1, W * 2))
        return _pack_bayer(noise)
    else:
        raise NotImplementedError()
