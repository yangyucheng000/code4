import mindspore as ms
import torch
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.amp import DynamicLossScaler
from mindspore.common.initializer import initializer
from mindspore.train.amp import auto_mixed_precision

from core.callbacks import MeanIoU
from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point
from core.schedulers import cosine_schedule_with_warmup
from torchsparse_ms import PointTensor, SparseTensor
import torchsparse_ms.nn.functional as F
from torchsparse_ms import nn as spnn
import torchsparse_ms

class CrossEntropyLossWithIgnored(nn.Cell):

    def __init__(self, sparse=False, reduction='none', ignore_index=255):
        super(CrossEntropyLossWithIgnored, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction=reduction)

    def construct(self, logits, labels):
        valid_index_trans = (labels != self.ignore_index).astype(ms.int32)
        valid_index = valid_index_trans.nonzero().flatten()

        # print(f"loss.valid_index: {valid_index}")
        # print(f"loss.valid_index.shape: {valid_index.shape}, loss.valid_index.dtype: {valid_index.dtype}")

        ce = self.ce(logits[valid_index], labels[valid_index])
        print('=============ce: %f ================' % ce)
        return ce

def main():
    import argparse
    import random
    import sys
    import os

    import mindspore as ms
    import mindspore.dataset as ds
    from mindspore import context, set_seed, Model, ParallelMode
    from mindspore.communication import init, get_rank, get_group_size
    from core.utils.config import configs
    from core.utils.local_adapter import execute_distributed, distributed
    from core.trainers import CustomWithLossCell

    import numpy as np
    from core import builder

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    configs.update(vars(args))

    # context.set_context(device_target='GPU')
    print(f'configs: {configs}')
    print("-->GPU数量: ", configs.n_gpus)
    rank = int(os.getenv('RANK_ID', '0'))
    if configs.n_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        init()
        rank = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        # is_distributed = True
        execute_distributed()
        # if rank == 0:
        #     recorder = Recorder(settings, settings.save_path)
        # else:
        #     recorder = 0
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU",
                            device_id=int(configs.gpu[0]))
        # is_distributed = False
        # recorder = Recorder(settings, settings.save_path)

    # if args.run_dir is None:
    #     args.run_dir = auto_set_run_dir()
    # else:
    #     set_run_dir(args.run_dir)

    # logger.info(' '.join([sys.executable] + sys.argv))
    # logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = np.random.randint(np.int32) % (2 ** 32 - 1)

    # seed = configs.train.seed + dist.rank(
    # ) * configs.workers_per_gpu * configs.num_epochs
    seed = configs.train.seed + rank * configs.workers_per_gpu * configs.num_epochs
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    # configs.dataset.name = 'dummy_kitti'
    # print('=====================using dummpy_kitti==================================')
    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        if distributed:
            rank_size = get_group_size()
            sampler = ms.dataset.DistributedSampler(
                num_shards=rank_size,
                shard_id=rank,
                shuffle=(split == 'train'),
            )
            dataflow[split] = ds.GeneratorDataset(
                dataset[split],
                column_names=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                sampler=sampler
            )
            dataflow[split] = dataflow[split].batch(
                batch_size=configs.batch_size,
                num_parallel_workers=configs.workers_per_gpu,
                per_batch_map=dataset[split].per_batch_map,
                # output_columns=['lidar', 'targets', 'targets_mapped', 'inverse_map', 'file_name']
                output_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name', 'num_vox',
                                'num_pts']
            )
        else:
            dataflow[split] = ds.GeneratorDataset(
                dataset[split],
                column_names=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                shuffle=False
            )
            dataflow[split] = dataflow[split].batch(
                batch_size=configs.batch_size,
                num_parallel_workers=configs.workers_per_gpu,
                per_batch_map=dataset[split].per_batch_map,
                # input_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                output_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name', 'num_vox',
                                'num_pts']
                # output_columns=['feed_dict_list']
            )

    # from visualize import visualize_pcd
    # for fd_tuple in dataflow['train'].create_tuple_iterator(output_numpy=True):
    #     feed_dict = dataset['train'].collate_fn(*fd_tuple)
    #     lidar = feed_dict['lidar']
    #     num_vox = [80000, 80000]
    #     targets = ms.ops.cast(feed_dict['targets'].F, ms.int64)
    #     cur = 0
    #     for n in num_vox:
    #         pts = lidar.F[cur:cur+n, :3]
    #         label = targets[cur:cur+n]
    #         visualize_pcd(xyz=pts, target=label)
    #         cur += n

    net = builder.make_model()
    # for i, tp in enumerate(net.trainable_params()):
    #     print("{:<20} | {}".format(i, tp.shape))
    # for p in net.parameters_and_names():
    #     print("{:<30} | {}".format(p[0], p[1].shape))
    # exit()

    # net = auto_mixed_precision(net, 'O2')
    dy_lr_list = cosine_schedule_with_warmup(configs)
    ce_ops = CrossEntropyLossWithIgnored(sparse=True, reduction='mean', ignore_index=255)
    # loss_scaler = DynamicLossScaler(scale_value=2 ** 10, scale_factor=2, scale_window=50)
    optimizer = builder.make_optimizer(net, dy_lr_list)
    max_miou = 0

    # net = SPVCNN_MS()
    def forward_fn(x, y):
        output = net(x)
        ce = ce_ops(output, y)
        # ce = loss_scaler.scale(ce)
        return ce

    def train_loop(epoch_idx):
        num_batches = dataflow['train'].get_dataset_size()
        net.set_train()
        for batch_idx, fd_tuple in enumerate(dataflow['train'].create_tuple_iterator(output_numpy=True)):
            feed_dict = dataset['train'].collate_fn(*fd_tuple)
            lidar = feed_dict['lidar']
            targets = feed_dict['targets'].F.astype(ms.int64)
            grad_fn = ms.value_and_grad(forward_fn, None, weights=net.trainable_params())
            loss, grad = grad_fn(lidar, targets)
            # loss = loss_scaler.unscale(loss)
            optimizer(grad)
            # print(f'-------------------after opitmizer-----------------\n')
            isexit = False
            for i, grad_sig in enumerate(grad):
                if (np.isnan(grad_sig.asnumpy()).any()):  # Checking for nan values
                    print("np.isnan name: ", i,
                          '\n-------------------------------------------\n',
                          grad_sig)  # Prints an index of samples with nan values
                    isexit = True
            if isexit:
                exit()

            result_line = 'Train: Epoch: [{0}][{1}/{2}]\t' \
                          'Loss: {loss} LR: {lr}\n'.format(
                epoch_idx, batch_idx, num_batches,
                loss=loss, lr=optimizer.get_lr()
            )
            print(result_line)

    def test_loop(epoch_idx):
        num_batches = dataflow['test'].get_dataset_size()
        net.set_train(False)
        test_loss = 0
        miou = MeanIoU(configs.data.num_classes, configs.data.ignore_label)
        for batch_idx, fd_tuple in enumerate(dataflow['test'].create_tuple_iterator(output_numpy=True)):
            feed_dict = dataset['train'].collate_fn(*fd_tuple)
            lidar = feed_dict['lidar']
            targets = feed_dict['targets'].F.astype(ms.int64)

            output = net(lidar)
            test_loss += ce_ops(output, targets)
            miou.update(output, targets)

        test_loss /= num_batches
        miou, ious = miou.get_iou()
        print(f'ious: {ious}')
        print(f'Test: Epoch: [{epoch_idx}] mIOU: {(100 * miou):>0.1f}%, Avg Loss: {test_loss} \n')
        return miou

    for epoch_idx in range(configs.num_epochs):
        train_loop(epoch_idx)
        miou = test_loop(epoch_idx)
        if max_miou < miou:
            max_miou = miou
            ms.save_checkpoint(net, "model.ckpt")
            print(f'save model weight successfully')

    print('Done !')

# if __name__ == '__main__':
#     x1 = Tensor(np.random.rand(92292, 4), dtype=ms.float32)
#     c1 = Tensor(np.random.randint(0, 1e5, size=(92292, 4)), dtype=ms.int32)
#     y = Tensor(np.random.randint(0, 19, size=(92292,)), dtype=ms.int64)
#     print(f"x1.shape:{x1.shape}, x1.dtype:{x1.dtype}")
#     print(f"c1.shape:{c1.shape}, c1.dtype:{c1.dtype}")
#
#     spvnas_test = SPVCNN_MS()
#     input = SparseTensor(x1, c1)
#     grad_fn = ms.value_and_grad(forward_fn, None, weights=spvnas_test.trainable_params())
#     loss, grad = grad_fn(input, y)
#     print(grad)

if __name__ == '__main__':
    main()




