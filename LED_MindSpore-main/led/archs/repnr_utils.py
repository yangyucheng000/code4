from copy import deepcopy
import math
# import torch
# from torch import nn
# from torch.nn import init
# from torch.nn import functional as F
# from torch.nn.init import kaiming_normal_, kaiming_uniform_
from torch.nn.modules.utils import _reverse_repeat_tuple

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer

def zero_init_(x, a=None):
    x.zero_()

def build_repnr_arch_from_base(base_arch, **repnr_kwargs):
    base_arch = deepcopy(base_arch)
    dont_convert_module = [] if 'dont_convert_module' not in repnr_kwargs \
                             else repnr_kwargs.pop('dont_convert_module')

    def recursive_converter(base_arch, **repnr_kwargs):
        if isinstance(base_arch, nn.Conv2d):
            return RepNRConv2d(base_arch, **repnr_kwargs)

        repnr_arch = deepcopy(base_arch)
        # for n, c in base_arch.named_children():
        #     if n not in dont_convert_module:
        #         setattr(repnr_arch, n, recursive_converter(c, **repnr_kwargs))

        for m in base_arch.cells_and_names():
            if m[0] not in dont_convert_module:
                setattr(repnr_arch, m[0], recursive_converter(m[1], **repnr_kwargs))

        return repnr_arch

    repnr_arch = recursive_converter(base_arch, **repnr_kwargs)
    return RepNRBase(base_arch, repnr_arch)


# class RepNRConv2d(nn.Module):
class RepNRConv2d(nn.Cell):
    # def _init_conv(self, init_type='kaiming_uniform_'):
    def _init_conv(self, init_type='HeUniform'):
        super().__init__()
        # weight = torch.zeros_like(self.main_weight)
        weight = ops.zeros_like(self.main_weight)
        init_func = eval(init_type)
        init_func(weight, a=math.sqrt(5))
        # fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        fan_in, _ = initializer._calculate_fan_in_and_fan_out(weight)

        # bias = torch.zeros((weight.size(0), ),
        #                     dtype=self.main_weight.dtype,
        #                     device=self.main_weight.device)

        bias = ops.zeros((weight.size(0), ),
                            dtype=self.main_weight.dtype,
                            )
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            # init.uniform_(bias, -bound, bound)
            bias = ops.uniform(bias.shape, -bound, bound)
        return weight, bias

    def _init_from_conv2d(self, conv2d: nn.Conv2d):
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.stride = conv2d.stride
        self.padding = _reverse_repeat_tuple(conv2d.padding, 2)
        self.padding_mode = 'constant' if conv2d.padding_mode == 'zeros' else conv2d.padding_mode
        self.dilation = conv2d.dilation
        self.groups = conv2d.groups
        self.bias = conv2d.bias is not None

        main_weight = conv2d.weight.data.clone()
        main_bias = conv2d.bias.data.clone() if self.bias else None
        self.main_weight = nn.Parameter(main_weight, requires_grad=True)
        self.main_bias = nn.Parameter(main_bias, requires_grad=True) if main_bias is not None else None

    def _init_alignments(self, align_opts):
        align_init_weight = align_opts.get('init_weight', 1.0)
        if 'init_bias' in align_opts:
            align_init_bias = align_opts['init_bias']
            align_bias = True
        else:
            align_bias = False
        align_channles = self.in_channels

        self.align_weights = nn.ParameterList(
            [
                # nn.Parameter(torch.ones((1, align_channles, 1, 1),
                #                         dtype=self.main_weight.dtype,
                #                         device=self.main_weight.device) * align_init_weight,
                #              requires_grad=True)

                nn.Parameter(ops.ones((1, align_channles, 1, 1),
                                        dtype=self.main_weight.dtype,
                                        ) * align_init_weight,
                             requires_grad=True)
                for _ in range(self.branch_num + 1)
            ]
        )
        self.align_biases = nn.ParameterList(
            [
                # nn.Parameter(torch.ones((1, align_channles, 1, 1),
                #                         dtype=self.main_weight.dtype,
                #                         device=self.main_weight.device) * align_init_bias,
                #             requires_grad=True)

                nn.Parameter(ops.ones((1, align_channles, 1, 1),
                                        dtype=self.main_weight.dtype,
                                        ) * align_init_bias,
                            requires_grad=True)
                for _ in range(self.branch_num + 1)
            ]
        ) if align_bias else None

    def _init_aux_conv(self, aux_conv_opts):
        if 'init' not in aux_conv_opts:
            # aux_conv_opts['init'] = 'kaiming_normal_'
            aux_conv_opts['init'] = 'HeNormal'


        aux_weight, aux_bias = self._init_conv(init_type=aux_conv_opts['init'])
        self.aux_weight = nn.Parameter(aux_weight, requires_grad=True)
        # self.aux_bias = nn.Parameter(torch.zeros_like(aux_bias), requires_grad=True) \
        #                 if aux_conv_opts.get('bias', False) else torch.zeros_like(aux_bias)
        self.aux_bias = nn.Parameter(ops.zeros_like(aux_bias), requires_grad=True) \
                        if aux_conv_opts.get('bias', False) else ops.zeros_like(aux_bias)

    def __init__(self, conv2d, branch_num,
                       align_opts, aux_conv_opts=None,
                       forward_type='reparameterize'):
        super().__init__()
        self.branch_num = branch_num
        self.forward_type = forward_type
        self.align_opts = align_opts
        self.aux_conv_opts = aux_conv_opts

        self._init_from_conv2d(conv2d)
        self._init_alignments(align_opts)
        if aux_conv_opts:
            self._init_aux_conv(aux_conv_opts)

        self.cur_branch = -1
        self.forward = self._reparameterize_forward if forward_type == 'reparameterize' \
            else self._trivial_forward

    def switch_forward_type(self, *, trivial=False, reparameterize=False):
        assert not (trivial and reparameterize)
        if trivial:
            self.forward = self._trivial_forward
            self.forward_type = 'trivial'
        elif reparameterize:
            self.forward = self._reparameterize_forward
            self.forward_type = 'reparameterize'

    @staticmethod
    def _sequential_reparamterize(k1, b1, k2, b2):
        # k1, b1 is the weight and bias of alignment
        def depthwise_to_normal(k, padding):
            k = k.reshape(-1)
            # k = torch.diag(k).unsqueeze(-1).unsqueeze(-1)
            k = ops.diag(k).unsqueeze(-1).unsqueeze(-1)
            # k = F.pad(k, _reverse_repeat_tuple(padding, 2), mode='constant', value=0.0)
            k = ops.pad(k, _reverse_repeat_tuple(padding, 2), mode='constant', value=0.0)
            return k

        def bias_pad(b, padding):
            # return F.pad(b, _reverse_repeat_tuple(padding, 2), mode='replicate')
            return ops.pad(b, _reverse_repeat_tuple(padding, 2), mode='replicate')


        padding = (k2.shape[-2] - 1) // 2, (k2.shape[-1] - 1) // 2
        k1 = depthwise_to_normal(k1, padding)
        # k = F.conv2d(k2, k1, stride=1, padding='same')
        # b = F.conv2d(bias_pad(b1, padding), k2, bias=b2, stride=1).reshape(-1)
        k = ops.conv2d(k2, k1, stride=1, padding='same')
        b = ops.conv2d(bias_pad(b1, padding), k2, bias=b2, stride=1).reshape(-1)
        return k, b

    @staticmethod
    def _parallel_reparamterize(k1, b1, k2, b2):
        return k1 + k2, b1 + b2

    @property
    def _weight_and_bias(self):
        index = self.cur_branch
        align_weight = self.align_weights[index]
        # align_bias = self.align_biases[index] if self.align_biases is not None \
        #     else torch.zeros_like(align_weight)
        align_bias = self.align_biases[index] if self.align_biases is not None \
            else ops.zeros_like(align_weight)

        main_weight, main_bias = self._sequential_reparamterize(
            align_weight, align_bias, self.main_weight, self.main_bias)

        if hasattr(self, 'aux_weight'):
            main_weight, main_bias = self._parallel_reparamterize(
                main_weight, main_bias, self.aux_weight, self.aux_bias)

        return main_weight, main_bias

    def _reparameterize_forward(self, x):
        weight, bias = self._weight_and_bias
        # x = F.pad(x, self.padding, self.padding_mode, value=0.0)
        # x = F.conv2d(x, weight, bias, stride=self.stride)
        x = ops.pad(x, self.padding, self.padding_mode, value=0.0)
        x = ops.conv2d(x, weight, bias, stride=self.stride)
        return x

    def _trivial_forward(self, x):
        index = self.cur_branch
        align_weight = self.align_weights[index]
        # align_bias = self.align_biases[index] if self.align_biases is not None \
        #     else torch.zeros_like(align_weight)
        align_bias = self.align_biases[index] if self.align_biases is not None \
            else ops.zeros_like(align_weight)

        aligned_x = x * align_weight + align_bias
        # padded_aligned_x = F.pad(aligned_x, self.padding, self.padding_mode, value=0.0)
        # main_x = F.conv2d(padded_aligned_x, self.main_weight, self.main_bias, stride=self.stride)
        padded_aligned_x = ops.pad(aligned_x, self.padding, self.padding_mode, value=0.0)
        main_x = ops.conv2d(padded_aligned_x, self.main_weight, self.main_bias, stride=self.stride)

        if hasattr(self, 'aux_weight'):
            # padded_x = F.pad(x, self.padding, self.padding_mode, value=0.0)
            # aux_x = F.conv2d(padded_x, self.aux_weight, self.aux_bias, stride=self.stride)
            padded_x = ops.pad(x, self.padding, self.padding_mode, value=0.0)
            aux_x = ops.conv2d(padded_x, self.aux_weight, self.aux_bias, stride=self.stride)
            main_x = main_x + aux_x

        return main_x

    def _trivial_forward_with_intermediate_features(self, x):
        index = self.cur_branch
        features = {}
        align_weight = self.align_weights[index]
        # align_bias = self.align_biases[index] if self.align_biases is not None \
        #     else torch.zeros_like(align_weight)
        align_bias = self.align_biases[index] if self.align_biases is not None \
            else ops.zeros_like(align_weight)

        features['in_feat'] = x
        aligned_x = x * align_weight + align_bias
        features['aligned_feat'] = aligned_x

        # padded_aligned_x = F.pad(aligned_x, self.padding, self.padding_mode, value=0.0)
        # main_x = F.conv2d(padded_aligned_x, self.main_weight, self.main_bias, stride=self.stride)
        padded_aligned_x = ops.pad(aligned_x, self.padding, self.padding_mode, value=0.0)
        main_x = ops.conv2d(padded_aligned_x, self.main_weight, self.main_bias, stride=self.stride)

        features['main_feat'] = main_x

        if hasattr(self, 'aux_weight'):
            # padded_x = F.pad(x, self.padding, self.padding_mode, value=0.0)
            # aux_x = F.conv2d(padded_x, self.aux_weight, self.aux_bias, stride=1)
            padded_x = ops.pad(x, self.padding, self.padding_mode, value=0.0)
            aux_x = ops.conv2d(padded_x, self.aux_weight, self.aux_bias, stride=1)
            features['aux_feat'] = aux_x
            main_x = main_x + aux_x

        features['out_feat'] = aux_x
        self.current_intermediate_features = features
        return main_x

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def extra_repr(self) -> str:
        s = (
            '{in_channels}, {out_channels}, kernel_size={kernel_size}, '
            'stride={stride}, padding={padding}, padding_mode={padding_mode},\n'
            'branch_num={branch_num}, align_opts={align_opts}\n'
            'forward_type={forward_type}'
        )
        if hasattr(self, 'aux_weight'):
            s += ', aux_conv_opts={aux_conv_opts}'
        return s.format(**self.__dict__)


# class RepNRBase(nn.Module):
class RepNRBase(nn.Cell):
    def __init__(self, arch, repnr_arch) -> None:
        super().__init__()
        self.base_module = arch
        self.repnr_module = repnr_arch

        # base_module do not need optimize
        for p in self.base_module.parameters():
            p.requires_grad_(False)

    def state_dict(self, *args, **kwargs):
        return self.repnr_module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.repnr_module.load_state_dict(state_dict, strict)
        return self

    def set_cur_branch(self, idx):
        for m in self.modules():
            if isinstance(m, RepNRConv2d):
                m.cur_branch = idx

    def forward(self, x, idx=-1):
        self.set_cur_branch(idx)
        return self.repnr_module(x)

    def switch_forward_type(self, *, trivial=False, reparameterize=False):
        for m in self.modules():
            if isinstance(m, RepNRConv2d):
                m.switch_forward_type(trivial=trivial, reparameterize=reparameterize)

    def generalize(self):
        def generalize_align_conv(align_conv: RepNRConv2d):
            align_weights = align_conv.align_weights[:-1]
            align_biases = align_conv.align_biases[:-1]
            average_weight = sum(align_weights) / len(align_weights)
            average_bias = sum(align_biases) / len(align_biases)
            align_conv.align_weights[-1].data = average_weight
            align_conv.align_biases[-1].data = average_bias

        for m in self.modules():
            if isinstance(m, RepNRConv2d):
                generalize_align_conv(m)

    @staticmethod
    def _set_requires_grad(align_conv: RepNRConv2d, *, pretrain=True, finetune=False, aux=True):
        assert not (pretrain and finetune)
        for i in range(align_conv.branch_num):
            align_conv.align_weights[i].requires_grad_(pretrain)
            if align_conv.align_biases is not None:
                align_conv.align_biases[i].requires_grad_(pretrain)
        align_conv.align_weights[-1].requires_grad_(finetune)
        if align_conv.align_biases is not None:
            align_conv.align_biases[-1].requires_grad_(finetune)

        # main weight and bias
        align_conv.main_weight.requires_grad_(pretrain)
        if align_conv.main_bias is not None:
            align_conv.main_bias.requires_grad_(pretrain)

        # aux weight and bias
        if hasattr(align_conv, 'aux_weight') and aux:
            align_conv.aux_weight.requires_grad_(finetune)
            if isinstance(align_conv.aux_bias, nn.Parameter):
                align_conv.aux_bias.requires_grad_(finetune)

    def pretrain(self):
        for m in self.modules():
            if isinstance(m, RepNRConv2d):
                self._set_requires_grad(m, pretrain=True, finetune=False, aux=False)

    def finetune(self, *, aux=False):
        for m in self.modules():
            if isinstance(m, RepNRConv2d):
                self._set_requires_grad(m, pretrain=False, finetune=True, aux=aux)

    def __repr__(self):
        return f'{self._get_name()}: {str(self.repnr_module)}'

    # @torch.no_grad()
    def deploy(self):
        def has_repnr_conv(m):
            for mm in m.modules():
                if isinstance(mm, RepNRConv2d):
                    return True
            return False

        def deploy(m_bak, m):
            if isinstance(m, RepNRConv2d):
                weight, bias = m._weight_and_bias
                m_bak.weight.data = weight
                m_bak.bias.data = bias
                return
            if not has_repnr_conv(m):
                m_bak.load_state_dict(m.state_dict())
                return

            for n, mm in m_bak.named_children():
                deploy(mm, getattr(m, n))

        self.set_cur_branch(-1)
        deploy(self.base_module, self.repnr_module)

        return self.base_module
