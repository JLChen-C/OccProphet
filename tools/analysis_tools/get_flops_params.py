# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config, DictAction
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

from mmdet3d.models import build_model
from tools.misc.fuse_conv_bn import fuse_module
from projects.occ_plugin import *

import time
import mmcv
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import (conv_flops_counter_hook, relu_flops_counter_hook,
                                          pool_flops_counter_hook, norm_flops_counter_hook,
                                          linear_flops_counter_hook, upsample_flops_counter_hook,
                                          deconv_flops_counter_hook)

import numpy as np
import torch.nn as nn
import torchvision
# from fvcore.nn import FlopCountAnalysis

def deformconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel
    # 
    sample_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * 8

    active_elements_count = batch_size * int(np.prod(output_dims))

    # overall_conv_flops = conv_per_position_flops * active_elements_count
    overall_conv_flops = (conv_per_position_flops + sample_per_position_flops) * active_elements_count

    bias_flops = 0

    # if conv_module.bias is not None:

    #     bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)

def wmsa_flops_counter_hook(wmsa_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size_mul_num_windows = input.shape[0]
    window_size_square = input.shape[1]
    embed_dims = input.shape[2]

    attn_per_position_flops = 2 * window_size_square * embed_dims

    active_elements_count = batch_size_mul_num_windows

    overall_wmsa_flops = attn_per_position_flops * active_elements_count

    wmsa_module.__flops__ += int(overall_wmsa_flops)

def get_modules_mapping():
    return {
        # convolutions
        nn.Conv1d: conv_flops_counter_hook,
        nn.Conv2d: conv_flops_counter_hook,
        mmcv.cnn.bricks.Conv2d: conv_flops_counter_hook,
        nn.Conv3d: conv_flops_counter_hook,
        mmcv.cnn.bricks.Conv3d: conv_flops_counter_hook,
        # activations
        nn.ReLU: relu_flops_counter_hook,
        nn.PReLU: relu_flops_counter_hook,
        nn.ELU: relu_flops_counter_hook,
        nn.LeakyReLU: relu_flops_counter_hook,
        nn.ReLU6: relu_flops_counter_hook,
        # poolings
        nn.MaxPool1d: pool_flops_counter_hook,
        nn.AvgPool1d: pool_flops_counter_hook,
        nn.AvgPool2d: pool_flops_counter_hook,
        nn.MaxPool2d: pool_flops_counter_hook,
        mmcv.cnn.bricks.MaxPool2d: pool_flops_counter_hook,
        nn.MaxPool3d: pool_flops_counter_hook,
        mmcv.cnn.bricks.MaxPool3d: pool_flops_counter_hook,
        nn.AvgPool3d: pool_flops_counter_hook,
        nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
        nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
        nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
        nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
        nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
        nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
        # normalizations
        nn.BatchNorm1d: norm_flops_counter_hook,
        nn.BatchNorm2d: norm_flops_counter_hook,
        nn.BatchNorm3d: norm_flops_counter_hook,
        nn.GroupNorm: norm_flops_counter_hook,
        nn.InstanceNorm1d: norm_flops_counter_hook,
        nn.InstanceNorm2d: norm_flops_counter_hook,
        nn.InstanceNorm3d: norm_flops_counter_hook,
        nn.LayerNorm: norm_flops_counter_hook,
        # FC
        nn.Linear: linear_flops_counter_hook,
        mmcv.cnn.bricks.Linear: linear_flops_counter_hook,
        # Upscale
        nn.Upsample: upsample_flops_counter_hook,
        # Deconvolution
        nn.ConvTranspose2d: deconv_flops_counter_hook,
        mmcv.cnn.bricks.ConvTranspose2d: deconv_flops_counter_hook,

        # mmcv.ops.DeformConv2d: deformconv_flops_counter_hook,
        # torchvision.ops.DeformConv2d: deformconv_flops_counter_hook
        mmcv.ops.DeformConv2dPack: deformconv_flops_counter_hook,
        WindowMSA: wmsa_flops_counter_hook
    }

mmcv.cnn.utils.flops_counter.get_modules_mapping = get_modules_mapping
print('add flops counter hook for deformable convolution')
print('add flops counter hook for windowed multi-head self-attention')
time.sleep(3)

def parse_args():
    parser = argparse.ArgumentParser(description='Get FLOPs and Params of a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[896, 1600],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='image',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cameras',
        type=int,
        default=6,
        help='number of cameras')
    parser.add_argument(
        '--frames',
        type=int,
        default=0,
        help='number of frames')
    # parser.add_argument(
    #     '--fuse-conv-bn',
    #     action='store_true',
    #     help='Whether to fuse conv and bn, this will slightly decrease the number of parameters')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    assert args.modality == 'image', 'currently only supports image input'

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')
    if args.cameras > 0:
        input_shape = (args.cameras, ) + input_shape
    if args.frames > 0:
        input_shape = (args.frames, ) + input_shape

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if hasattr(model, 'deploy'):
        model.deploy()
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)
    if torch.cuda.is_available():
        model.cuda()
    # model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))

    # flops, params = get_model_complexity_info(model, input_shape)
    # split_line = '=' * 30
    # print(f'{split_line}\nInput shape: {input_shape}\n'
    #       f'Flops: {flops}\nParams: {params}\n{split_line}')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')
    
    # model.train()
    # params = sum(p.numel() for p in model.parameters()) / 1024 ** 2
    # model.eval()
    # dummy_input = torch.randn((1, *input_shape), device=next(model.parameters()).device)
    with torch.no_grad():
        # flops = FlopCountAnalysis(model, dummy_input).total() / 1024 ** 3
        flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 32
    # print(f'{split_line}\nInput shape: {input_shape}\n'
    #       f'Flops: {flops:.3f} G\nParams: {params:.3f} M\n{split_line}')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')


if __name__ == '__main__':
    main()
