# Copyright (c) OpenMMLab. All rights reserved.
import os
import argparse
import time
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

from mmdet3d.models import build_model
from tools.misc.fuse_conv_bn import fuse_module
from projects.occ_plugin import *

def parse_args():
    parser = argparse.ArgumentParser(description='Get FLOPs and Params of a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
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
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='samples to benchmark')
    parser.add_argument(
        '--log-interval',
        default=50,
        type=int,
        help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to use fp16 for inference, this will slightly increase the inference speed')
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
    if args.fp16:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    if hasattr(model, 'deploy'):
        model.deploy()
    if args.fuse_conv_bn:
        model = fuse_module(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.fp16:
        model.half()
    model.eval()
    device = next(model.parameters()).device

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))

    with torch.no_grad():
        num_warmup = 5
        for i in range(num_warmup):
            dummy_input = torch.randn((1, *input_shape), device=device)
            if args.fp16:
                dummy_input = dummy_input.half()
            model(dummy_input)

        infer_time = 0.
        for i in range(args.samples):
            dummy_input = torch.randn((1, *input_shape), device=device)
            if args.fp16:
                dummy_input = dummy_input.half()

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            model(dummy_input)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            infer_time += elapsed

            if (i + 1) % args.log_interval == 0:
                fps = (i + 1) / infer_time
                latency = infer_time / (i + 1) * 1000
                print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                    f'fps: {fps:.2f} sample / s || latency: {latency:.2f} ms / sample')

    fps = args.samples / infer_time
    latency = infer_time / args.samples * 1000
    print(f'Overall fps: {fps:.2f} sample / s || Overall latency: {latency:.2f} ms / sample')

if __name__ == '__main__':
    main()
