import numpy as np
import mindspore as ms
import mindspore.ops as ops
import torch

if __name__ == '__main__':
    sample = np.load("./ops_sum.npz")

    n_ms = ms.Tensor(sample['ms'], dtype=ms.float32)
    n_torch = torch.tensor(sample['torch'], dtype=torch.float32).cuda()

    print(f'unique(n_ms - n_torch): '
          f'{np.unique(n_ms.asnumpy() - n_torch.detach().cpu().numpy())}')

    n_ms_sum = ops.sum(n_ms, dim=0)
    n_torch_sum = torch.sum(n_torch, dim=0)

    print(f'unique(n_ms_sum - n_torch_sum): '
          f'{np.unique(n_ms_sum.asnumpy() - n_torch_sum.detach().cpu().numpy())}')
