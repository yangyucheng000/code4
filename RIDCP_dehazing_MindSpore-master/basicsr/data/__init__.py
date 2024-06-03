import importlib
import numpy as np
import random
# import torch
# import torch.utils.data
import mindspore as ms
from mindspore import dataset as ds

from copy import deepcopy
from functools import partial
from os import path as osp

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} ' 'is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        # dataloader_args = dict(
        #     dataset=dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=num_workers,
        #     sampler=sampler,
        #     drop_last=True)
            
        dataloader_args = dict(
            # source=dataset, # 参数名不同
            source=dataset, # 参数名不同
            # batch_size=batch_size, MindSpore通过 mindspore.dataset.batch 操作支持
            # shuffle=False, # 默认值不同
            # num_parallel_workers=num_workers, # 参数名不同 # 报错 ValueError: num_parallel_workers exceeds the boundary between 1 and 12!
            sampler=sampler, # 一致
            # drop_last=True # MindSpore通过 mindspore.dataset.batch 操作支持
            )    
        
        if sampler is None:
            dataloader_args['shuffle'] = True
            
# MindSpore不支持
        # dataloader_args['worker_init_fn'] = partial(
        #     worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        
    elif phase in ['val', 'test']:  # validation
        batch_size = dataset_opt.get('batch_size_per_gpu', 1)
        # dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
        dataloader_args = dict(
            source=dataset, 
            # batch_size=1, MindSpore通过 mindspore.dataset.batch 操作支持
            shuffle=False, 
            # num_workers=0
            num_parallel_workers=1
            )

    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

# MindSpore不支持
    # dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
# MindSpore通过 create_tuple_iterator 的 num_epoch 参数支持
    # dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)
    persistent_workers = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    # if prefetch_mode == 'cpu':  # CPUPrefetcher
    if prefetch_mode == 'CPU':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        # return torch.utils.data.DataLoader(**dataloader_args)
        dataloader = ds.GeneratorDataset(**dataloader_args, column_names=['data'])
        dataloader = dataloader.batch(batch_size=batch_size, drop_remainder=True)
        # 如果设置参数大于1则与 persistent_workers 为True一致
        if persistent_workers == True:
            persistent_workers = 2
        else:
            persistent_workers = -1
        dataloader = dataloader.create_tuple_iterator(num_epochs=persistent_workers)
        return dataloader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
