# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
# import torch
import mindspore as ms
# import torch.distributed as dist
from mindspore.communication import init, get_rank, get_group_size
# import torch.multiprocessing as mp
from mindspore import context
def init_dist(launcher, backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_mindspore(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def set_device(rank, num_gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % num_gpus)

# def _init_dist_pytorch(backend, **kwargs):
def _init_dist_mindspore(backend):  
    rank = int(os.environ['RANK'])
    # num_gpus = torch.cuda.device_count()
    num_gpus = get_group_size()
    set_device(rank, num_gpus)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU") # PyNative模式不支持并行
    # dist.init_process_group(backend=backend, **kwargs)
    init(backend_name=backend)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    # num_gpus = torch.cuda.device_count()
    num_gpus = get_group_size()
    # torch.cuda.set_device(proc_id % num_gpus)
    set_device(proc_id, num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    # dist.init_process_group(backend=backend)
    init(backend_name=backend)

def get_dist_info():
    # if dist.is_available():
    #     initialized = dist.is_initialized()
    # else:
    #     initialized = False
    # if initialized:
        # rank = dist.get_rank()
    try:
        rank = get_rank()
        # world_size = dist.get_world_size()
        world_size = get_group_size()
    # else:
    except RuntimeError:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
# import torch
import mindspore as ms
# import torch.distributed as dist
from mindspore.communication import init, get_rank, get_group_size
# import torch.multiprocessing as mp
from mindspore import context
def init_dist(launcher, backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_mindspore(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def set_device(rank, num_gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % num_gpus)

# def _init_dist_pytorch(backend, **kwargs):
def _init_dist_mindspore(backend):  
    rank = int(os.environ['RANK'])
    # num_gpus = torch.cuda.device_count()
    num_gpus = get_group_size()
    set_device(rank, num_gpus)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU") # PyNative模式不支持并行
    # dist.init_process_group(backend=backend, **kwargs)
    init(backend_name=backend)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    # num_gpus = torch.cuda.device_count()
    num_gpus = get_group_size()
    # torch.cuda.set_device(proc_id % num_gpus)
    set_device(proc_id, num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    # dist.init_process_group(backend=backend)
    init(backend_name=backend)

def get_dist_info():
    # if dist.is_available():
    #     initialized = dist.is_initialized()
    # else:
    #     initialized = False
    # if initialized:
        # rank = dist.get_rank()
    try:
        rank = get_rank()
        # world_size = dist.get_world_size()
        world_size = get_group_size()
    # else:
    except RuntimeError:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
