import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torch.utils.checkpoint import checkpoint as CP

from mmcv.cnn.bricks import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmcv.ops import DeformConv2d
from mmdet.models.utils import make_divisible
from mmdet3d.models.builder import NECKS, MIDDLE_ENCODERS, build_middle_encoder

from .attention import ShiftWindowMSA


class TemporalAttention(nn.Module):
    
    def __init__(self,
                 embed_dims,
                 num_heads,
                 timesteps=None,
                 with_pe=True):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.with_pe = with_pe

        if with_pe:
            self.temporal_pe = nn.Parameter(torch.randn(timesteps, embed_dims))

        self.pre_norm = nn.LayerNorm(embed_dims)
        self.attn = MultiheadAttention(
            embed_dims, num_heads, batch_first=True)
        self.post_norm = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, feedforward_channels=4 * embed_dims)
        self.out_norm = nn.LayerNorm(embed_dims)

    @staticmethod
    def create_temporal_attn_mask(x):
        """
        In nn.MultiheadAttention:

        For a binary mask, a ``True`` value indicates that the corresponding position is not allowed to attend.
        For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend.
        For a float mask, the mask values will be added to the attention weight.
        """
        if x.ndim == 4:
            B, T, N, C = [int(ss) for ss in x.shape]
        else:
            B, T, C = [int(ss) for ss in x.shape]
        attn_mask = x.new_ones((T, T), dtype=torch.bool)
        attn_mask = attn_mask.tril()
        return ~attn_mask
    
    def _check_shape(self, name, value):
        ndim = value.ndim
        assert ndim in [3, 4], f'requires {name} in 3-dim (B, T, C) or 4-dim (B, T, N, C), got shape: {value.shape}'

    def forward(self, query, key=None, value=None, attn_mask=None):
        ndim = query.ndim
        # assert ndim in [3, 4], f'requires query in 3-dim (B, T, C) or 4-dim (B, T, N, C), got shape: {query.shape}'
        self._check_shape('query', query)
        if key is not None:
            self._check_shape('key', key)
        if value is not None:
            self._check_shape('value', value)
        if ndim == 4:
            B, Tq, N, C = [int(ss) for ss in query.shape]

        if ndim == 4:
            # (B, Tq, N, C) -> (B, N, Tq, C) -> (B*N, Tq, C)
            query = query.transpose(1, 2).flatten(0, 1)

            # (B, Tk, N, C) -> (B, N, Tk, C) -> (B*N, Tk, C)
            if key is not None:
                key = key.transpose(1, 2).flatten(0, 1)
            # (B, Tv, N, C) -> (B, N, Tv, C) -> (B*N, Tv, C)
            if value is not None:
                value = value.transpose(1, 2).flatten(0, 1)

        if self.with_pe:
            # (B/B*N, Tq, C) + (1, Tq, C) -> (B/B*N, Tq, C)
            query = query + self.temporal_pe[None, :, :]

        query_norm = self.pre_norm(query)
        if key is None:
            key = query_norm
        if value is None:
            value = query_norm
        query = query + self.attn(query=query_norm, key=key, value=value, attn_mask=attn_mask)
        query = query + self.ffn(self.post_norm(query))
        query = self.out_norm(query)
        if ndim == 4:
            # (B*N, Tq, C) -> (B, N, Tq, C) -> (B, Tq, N, C)
            query = query.view(B, -1, Tq, C).transpose(1, 2).contiguous()
        return query


@MIDDLE_ENCODERS.register_module()
class HybridFusion(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 timesteps,
                 height_kernel_size=1,
                 residual=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        
        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.timesteps = timesteps
        self.residual = residual

        self.scene_branch = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            ConvModule(embed_dims, embed_dims, kernel_size=1,
                conv_cfg=dict(type='Conv3d'), norm_cfg=norm_cfg))
        self.scene_temporal_attn = TemporalAttention(embed_dims, num_heads, timesteps)

        self.bev_pre_norm = nn.LayerNorm(embed_dims)
        self.bev_branch = ShiftWindowMSA(embed_dims, num_heads, window_size,
            pad_small_map=True, window_msa='WindowMSA')
        self.bev_ffn = FFN(embed_dims, feedforward_channels=4 * embed_dims)
        self.bev_post_norm = nn.LayerNorm(embed_dims)
        self.bev_out_norm = nn.LayerNorm(embed_dims)
        self.bev_temporal_attn = TemporalAttention(embed_dims, num_heads, timesteps)

        self.height_branch = ConvModule(embed_dims, embed_dims,
            kernel_size=height_kernel_size, padding=(height_kernel_size - 1) // 2,
            conv_cfg=dict(type='Conv1d'), norm_cfg=norm_cfg)
        self.height_temporal_attn = TemporalAttention(embed_dims, num_heads, timesteps)

    def forward(self, vox_feats):
        # vox_feats: (B, T, C, H, W, Z)
        assert vox_feats.ndim == 6, f'requires vox_feats in 6-dim (B, T, C, H, W, Z), got shape: {vox_feats.shape}'
        B, T, C, H, W, Z = [int(ss) for ss in vox_feats.shape]

        # (B, T, C, H, W, Z) -> (B*T, C, 1, 1, 1)
        scene_feats = self.scene_branch(vox_feats.view(-1, C, H, W, Z))
        # (B*T, C, 1, 1, 1) -> (B, T, C)
        scene_feats = scene_feats.view(B, T, C)
        temporal_attn_mask = TemporalAttention.create_temporal_attn_mask(scene_feats)
        scene_feats = self.scene_temporal_attn(scene_feats, attn_mask=temporal_attn_mask)

        # (B, T, C, H, W, Z) -> (B, T, C, H, W) -> (B, T, N(=H*W), C)
        bev_feats = vox_feats.mean(dim=-1).permute(0, 1, 3, 4, 2)
        # (B*T, N, C)
        bev_feats = bev_feats.view(-1, H*W, C)
        bev_feats = bev_feats + self.bev_branch(self.bev_pre_norm(bev_feats), hw_shape=[H, W])
        bev_feats = bev_feats + self.bev_ffn(self.bev_post_norm(bev_feats))
        bev_feats = self.bev_out_norm(bev_feats)
        # (B*T, N, C) -> (B, T, N, C)
        bev_feats = bev_feats.view(B, T, -1, C)
        bev_feats = self.bev_temporal_attn(bev_feats, attn_mask=temporal_attn_mask)

        # (B, T, C, H, W, Z) -> (B, T, C, Z) -> (B*T, C, Z) -> (B, T, C, Z)
        height_feats = self.height_branch(vox_feats.mean([-3, -2]).view(-1, C, Z)).view(B, T, C, Z).contiguous()
        # (B, T, C, Z) -> (B, T, Z, C)
        height_feats = height_feats.transpose(-2, -1)
        height_feats = self.height_temporal_attn(height_feats, attn_mask=temporal_attn_mask)

        # (B, T, N, C) + (B, T, 1, C) -> (B, T, N, C)
        bev_feats += scene_feats[:, :, None, :]

        # (B, T, N, C) -> (B, T, C, N) -> (B, T, C, H, W, 1)
        bev_feats = bev_feats.transpose(-2, -1).reshape(B, T, C, H, W, 1)
        # (B, T, Z, C) -> (B, T, C, Z) -> (B, T, C, 1, 1, Z)
        height_feats = height_feats.transpose(-2, -1)[:, :, :, None, None, :]

        output = bev_feats + height_feats
        if self.residual:
            output += vox_feats
        return output


@MIDDLE_ENCODERS.register_module()
class ConditionalGenerator(nn.Module):
    
    def __init__(self,
                 in_timesteps,
                 out_timesteps,
                 in_channels,
                 num_heads,
                 kernel_size=1,
                 use_ta=False,
                 norm_and_act=False,
                 with_cp=False,
                 leave_attn_unused=True, # this a legacy mistake, set it to False when training new model
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):

        super().__init__()

        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.use_ta = use_ta
        self.norm_and_act = norm_and_act
        self.with_cp = with_cp

        self.scene_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            ConvModule(in_channels, in_channels, kernel_size=1,
                conv_cfg=dict(type='Conv3d'), norm_cfg=norm_cfg))
        if use_ta:
            self.scene_temporal_attn = TemporalAttention(in_channels, num_heads, in_timesteps)
        elif leave_attn_unused:
            self.scene_temporal_attn = TemporalAttention(in_channels, num_heads, in_timesteps)

        self.pred_kernel = nn.Linear(
            in_timesteps * in_channels,
            out_timesteps * in_channels * in_timesteps * in_channels * kernel_size ** 3,)
            # kernel_size=kernel_size,
            # padding=(kernel_size - 1) // 2)

        if norm_and_act:
            self.norm = build_norm_layer(
                cfg=norm_cfg,
                num_features=out_timesteps * in_channels)[1]
            self.act = nn.ReLU(True)
        else:
            self.pred_bias = nn.Linear(
                in_timesteps * in_channels,
                out_timesteps * in_channels,)
                # kernel_size=kernel_size,
                # padding=(kernel_size - 1) // 2)

    def forward(self, vox_feats):
        # vox_feats: (B, T, C, H, W, Z)
        assert vox_feats.ndim == 6, f'requires vox_feats in 6-dim (B, T, C, H, W, Z), got shape: {vox_feats.shape}'
        B, T, C, H, W, Z = [int(ss) for ss in vox_feats.shape]
        T2 = self.out_timesteps

        def _forward(vox_feats): 

            # (B, T, C, H, W, Z) -> (B*T, C, 1, 1, 1)
            scene_feats = self.scene_extractor(vox_feats.view(-1, C, H, W, Z))
            if self.use_ta:
                # (B*T, C, 1, 1, 1) -> (B, T, C)
                scene_feats = scene_feats.view(B, T, C)
                scene_feats = self.scene_temporal_attn(scene_feats)

            # (B*T, C, 1, 1, 1)/(B, T, C) -> (B, T*C)
            scene_feats = scene_feats.view(B, -1)
            # (B, T2*C*T*C*k*k*k)
            kernel = self.pred_kernel(scene_feats)
            if not self.norm_and_act:
                # (B, T2*C)
                bias = self.pred_bias(scene_feats)
            else:
                bias = None

            # (B, T2*C*T*C*k*k*k) -> (B*T2*C, T*C, k, k, k)
            kernel = kernel.view(-1, T*C, self.kernel_size, self.kernel_size, self.kernel_size)
            if bias is not None:
                # (B, T2*C) -> (B*T2*C,)
                bias = bias.view(-1)

            # (1, B*T*C, H, W, Z) conv (B*T2*C, T*C, k, k, k) + (1, B*T2*C) -> (1, B*T2*C, H, W, Z)
            pred_vox_feats = F.conv3d(
                vox_feats.reshape(1, -1, H, W, Z), kernel, bias, padding=(self.kernel_size - 1)//2, groups=B)
            # # (1, B*T2*C, H, W, Z) -> (B, T2, C, H, W, Z)
            # pred_vox_feats = pred_vox_feats.view(B, -1, C, H, W, Z)

            # (1, B*T2*C, H, W, Z) -> (B, T2*C, H, W, Z)
            pred_vox_feats = pred_vox_feats.view(B, -1, H, W, Z)
            
            if self.norm_and_act:
                pred_vox_feats = self.norm(pred_vox_feats)
                pred_vox_feats = self.act(pred_vox_feats)
            # (B, T2*C, H, W, Z) -> (B, T2, C, H, W, Z)
            pred_vox_feats = pred_vox_feats.view(B, -1, C, H, W, Z)
            return pred_vox_feats

        if self.with_cp:
            return CP(_forward, vox_feats)
        else:
            return _forward(vox_feats)


@MIDDLE_ENCODERS.register_module()
class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 embed_dims,
                 out_channels=None,
                 in_proj_kernel_size=1,
                 downsample_layers=3, # size: 128->64->32->16
                 downsample_plugin_layers=None,
                 upsample_plugin_layers=None,
                 concat=False,
                 conv_after_readd=False,
                 multiscale_output=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 upsample_norm_cfgs=list(),
                 unpad_output=True):

        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.multiscale_output = multiscale_output
        self.unpad_output = unpad_output

        if in_channels != embed_dims:
            self.in_proj = ConvModule(
                in_channels, embed_dims,
                kernel_size=in_proj_kernel_size, padding=(in_proj_kernel_size - 1)//2,
                conv_cfg=dict(type='Conv3d'), norm_cfg=norm_cfg)
        else:
            self.in_proj = None

        # if (out_channels is not None) and (out_channels != embed_dims):
        if out_channels is not None:
            if multiscale_output:
                in_c = embed_dims * 2 ** downsample_layers
                out_proj = []
                for i in range(downsample_layers + 1):
                    out_proj.append(
                        ConvModule(
                            in_c,
                            out_channels,
                            kernel_size=1,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=norm_cfg)
                    )
                    in_c //= 2
                self.out_proj = nn.ModuleList(out_proj)
            else:
                self.out_proj = ConvModule(
                    embed_dims,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg)
        else:
            self.out_proj = None

        if downsample_plugin_layers is not None:
            assert len(downsample_plugin_layers) == downsample_layers
        if upsample_plugin_layers is not None:
            assert len(upsample_plugin_layers) == downsample_layers
        
        if downsample_plugin_layers is not None:
            self.downsample_plugin_layers = nn.ModuleList(
                build_middle_encoder(downsample_plugin_layers[i]) for i in range(len(downsample_plugin_layers)))
        else:
            self.downsample_plugin_layers = downsample_plugin_layers
        self.downsample_layers = nn.ModuleList()
        in_c = embed_dims
        for i in range(downsample_layers):
            self.downsample_layers.append(
                ConvModule(
                    in_c,
                    2 * in_c,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg
                ))
            in_c *= 2

        if upsample_plugin_layers is not None:
            self.upsample_plugin_layers = nn.ModuleList(
                build_middle_encoder(upsample_plugin_layers[i]) for i in range(len(upsample_plugin_layers)))
        else:
            self.upsample_plugin_layers = upsample_plugin_layers
        self.concat = concat
        self.conv_after_readd = conv_after_readd
        self.upsample_layers = nn.ModuleList()
        if len(upsample_norm_cfgs) == 0:
            upsample_norm_cfgs = [norm_cfg for i in range(downsample_layers)]
        if concat:
            self.merge_convs = nn.ModuleList()
        if (not concat) and conv_after_readd:
            self.upsample_convs = nn.ModuleList()
        for i in range(downsample_layers):
            if concat:
                self.upsample_layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                        ConvModule(
                            in_c,
                            in_c // 2,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=upsample_norm_cfgs[i]
                        )
                    )
                )
                self.merge_convs.append(
                    ConvModule(
                        in_c,
                        in_c // 2,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=upsample_norm_cfgs[i]
                    )
                )
            elif not conv_after_readd:
                self.upsample_layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                        ConvModule(
                            in_c,
                            in_c // 2,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=upsample_norm_cfgs[i]
                        )
                    )
                )
            else:
                self.upsample_layers.append(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
                )
                self.upsample_convs.append(
                    ConvModule(
                        in_c,
                        in_c // 2,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=upsample_norm_cfgs[i]
                    )
                )
            in_c //= 2

    def _adaptive_pad(self, vox_feats):
        B, C, H, W, Z = [int(ss) for ss in vox_feats.shape]
        divisor = 2 ** len(self.downsample_layers)
        if (H % divisor != 0) or (W % divisor != 0) or (Z % divisor != 0):
            padding = (
                0, make_divisible(Z, divisor) - Z,
                0, make_divisible(W, divisor) - W,
                0, make_divisible(H, divisor) - H
            )
            vox_feats = F.pad(vox_feats, padding)
        return vox_feats

    def forward(self, vox_feats):
        # vox_feats: (B, T, C, H, W, Z)
        assert vox_feats.ndim == 6, f'requires vox_feats in 6-dim (B, T, C, H, W, Z), got shape: {vox_feats.shape}'
        # B, T, C, H, W, Z = vox_feats.shape
        B, T, C_in, H_in, W_in, Z_in = [int(ss) for ss in vox_feats.shape]

        vox_feats_ = vox_feats.view(-1, C_in, H_in, W_in, Z_in)
        vox_feats_ = self._adaptive_pad(vox_feats_)

        # vox_feats_ = vox_feats
        if self.in_proj is not None:
            vox_feats_ = self.in_proj(vox_feats_)

        _, C, H, W, Z = [int(ss) for ss in vox_feats_.shape]
        n_layers = len(self.downsample_layers)

        # 128,128,10 -> 64,64,5 -> 32,32,2 -> 16,16,1 -
        #          +          +          +            |
        # 128,128,10 <- 64,64,5 <- 32,32,2 <- 16,16,1 -
        downsampled_vox_feats_list = [vox_feats_]
        for i in range(n_layers):
            vox_feats_ = self.downsample_layers[i](vox_feats_)
            if self.downsample_plugin_layers is not None:
                stride = 2 ** (i + 1)
                # stride = 2 ** i
                vox_feats_ = vox_feats_.view(B, T, stride*C, H//stride, W//stride, Z//stride)
                vox_feats_ = self.downsample_plugin_layers[i](vox_feats_)
                # downsampled_vox_feats_list.append(vox_feats_)
                vox_feats_ = vox_feats_.view(-1, stride*C, H//stride, W//stride, Z//stride)
            # vox_feats_ = self.downsample_layers[i](vox_feats_)
            if i != n_layers - 1:
                downsampled_vox_feats_list.append(vox_feats_)

        if self.multiscale_output:
            outputs = [vox_feats_]
        for i in range(n_layers):
            # vox_feats_ = self.upsample_layers[i](vox_feats_)
            # vox_feats_ = vox_feats_ + downsampled_vox_feats_list[-(i + 2)]
            if self.upsample_plugin_layers is not None:
                # stride = 2 ** (n_layers - i - 1)
                stride = 2 ** (n_layers - i)
                vox_feats_ = vox_feats_.view(B, T, stride*C, H//stride, W//stride, Z//stride)
                vox_feats_ = self.upsample_plugin_layers[i](vox_feats_)
                vox_feats_ = vox_feats_.view(-1, stride*C, H//stride, W//stride, Z//stride)
            vox_feats_ = self.upsample_layers[i](vox_feats_)
            if self.concat:
                vox_feats_ = torch.cat([vox_feats_, downsampled_vox_feats_list[-(i + 1)]], dim=1)
                vox_feats_ = self.merge_convs[i](vox_feats_)
            else:
                # vox_feats_ = vox_feats_ + downsampled_vox_feats_list[-(i + 2)]
                vox_feats_ = vox_feats_ + downsampled_vox_feats_list[-(i + 1)]
                if self.conv_after_readd:
                    vox_feats_ = self.upsample_convs[i](vox_feats_)
            if self.multiscale_output:
                outputs.append(vox_feats_)
        
        # height or width: 128, 64, 32, 16
        if self.multiscale_output:
            if self.unpad_output:
                spatial_sizes = [
                    [H_in//(2**(n_layers - i)), W_in//(2**(n_layers - i)), Z_in//(2**(n_layers - i))]
                    for i in range(n_layers + 1)]
                outputs = [
                    outputs[i][:, :, :int(spatial_sizes[i][0]), :int(spatial_sizes[i][1]), :int(spatial_sizes[i][2])]
                    for i in range(n_layers + 1)]
            if self.out_proj is not None:
                outputs = [self.out_proj[i](outputs[i]) for i in range(n_layers + 1)]
            if self.unpad_output:
                outputs = [
                    outputs[i].reshape(B, T, -1, spatial_sizes[i][0], spatial_sizes[i][1], spatial_sizes[i][2])
                    for i in range(n_layers + 1)]
            else:
                outputs = [
                    outputs[i].reshape(B, T, -1, *[int(ss) for ss in outputs[i].shape[-3:]])
                    for i in range(n_layers + 1)]
            return list(reversed(outputs))
        else:
            if self.unpad_output:
                vox_feats_ = vox_feats_[:, :, :int(H_in), :int(W_in), :int(Z_in)]
            if self.out_proj is not None:
                vox_feats_ = self.out_proj(vox_feats_)
            if self.unpad_output:
                vox_feats = vox_feats_.reshape(B, T, -1, H_in, W_in, Z_in)
            else:
                vox_feats = vox_feats_.reshape(B, T, -1, *[int(ss) for ss in vox_feats_.shape[-3:]])
            return vox_feats


@NECKS.register_module()
class UNetPredictor(nn.Module):

    def __init__(self,
                 pre_unet=None,
                 in_channels=None,
                 embed_dims=None,
                 in_timesteps=None,
                 out_timesteps=None,
                 post_unet=None,
                 pred_module=None,
                 depthwise=False,
                 kernel_size=1,
                 normact=False,
                 multiscale_output=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):

        super().__init__()

        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.multiscale_output = multiscale_output
        self.fp16_enabled = False
        
        self.pre_unet = build_middle_encoder(pre_unet) if pre_unet else None
        if pred_module is not None:
            self.pred_module = build_middle_encoder(pred_module)
            if self.pre_unet is None:
                assert in_channels is not None
                assert embed_dims is not None
                self.in_proj = ConvModule(
                    in_channels,
                    embed_dims,
                    kernel_size=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg)
        else:
            self.pred_module = None
            if self.pre_unet is None:
                assert in_channels is not None
                assert embed_dims is not None
                self.in_proj = ConvModule(
                    in_channels,
                    embed_dims,
                    kernel_size=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg)
                self.pred_linear = nn.Conv3d(
                    in_timesteps * embed_dims, out_timesteps * embed_dims,
                    kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            else:
                embed_dims = self.pre_unet.out_channels or self.pre_unet.embed_dims
                if normact:
                    if depthwise:
                        self.pred_linear = nn.Sequential(
                            ConvModule(
                                in_timesteps * embed_dims,
                                in_timesteps * embed_dims,
                                kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2,
                                groups=in_timesteps * embed_dims,
                                conv_cfg=dict(type='Conv3d'),
                                norm_cfg=norm_cfg),
                            ConvModule(
                                in_timesteps * embed_dims,
                                out_timesteps * embed_dims,
                                kernel_size=1,
                                conv_cfg=dict(type='Conv3d'),
                                norm_cfg=norm_cfg))
                    else:
                        self.pred_linear = ConvModule(
                            in_timesteps * embed_dims,
                            out_timesteps * embed_dims,
                            kernel_size=kernel_size,
                            padding=(kernel_size - 1) // 2,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=norm_cfg)
                else:
                    if depthwise:
                        self.pred_linear = nn.Sequential(
                            ConvModule(
                                in_timesteps * embed_dims,
                                in_timesteps * embed_dims,
                                kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2,
                                groups=in_timesteps * embed_dims,
                                conv_cfg=dict(type='Conv3d'),
                                norm_cfg=norm_cfg),
                            nn.Conv3d(
                                in_timesteps * embed_dims,
                                out_timesteps * embed_dims,
                                kernel_size=1))
                    else:
                        self.pred_linear = nn.Conv3d(
                            in_timesteps * embed_dims, out_timesteps * embed_dims,
                            kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        if post_unet:
            post_unet.update(multiscale_output=multiscale_output)
            self.post_unet = build_middle_encoder(post_unet)
        else:
            self.post_unet = None

    def forward(self, vox_feats, **kwargs):
        # vox_feats: (B, T, C, H, W, Z)
        assert vox_feats.ndim == 6, f'requires vox_feats in 6-dim (B, T, C, H, W, Z), got shape: {vox_feats.shape}'
        B, T, C, H, W, Z = [int(ss) for ss in vox_feats.shape]
        
        if self.pre_unet is not None:
            # (B, T, C, H, W, Z)
            vox_feats = self.pre_unet(vox_feats)
        else:
            vox_feats = self.in_proj(vox_feats.flatten(0, 1)).reshape(B, T, -1, H, W, Z)
        past_vox_feats_inter = vox_feats
        c, h, w, z = [int(ss) for ss in vox_feats.shape[-4:]]

        # (B, T2, C, H, W, Z)
        if self.pred_module is not None:
            pred_vox_feats = self.pred_module(vox_feats)
        else:
            pred_vox_feats = self.pred_linear(vox_feats.reshape(B, T*c, h, w, z)).view(B, self.out_timesteps, -1, h, w, z)
        future_vox_feats_inter = pred_vox_feats
        
        if self.post_unet is not None:
            # (B, T, C, H, W, Z) cat (B, T2, C, H, W, Z) -> (B, T+T2, C, H, W, Z)
            vox_feats = torch.cat([vox_feats, pred_vox_feats], dim=1)
            
            if self.multiscale_output:
                vox_feats_list = self.post_unet(vox_feats)
                future_vox_feats_list = [vox_feats_[:, self.in_timesteps:] for vox_feats_ in vox_feats_list]
                return [future_vox_feats_inter, future_vox_feats_list]
            else:
                # (B, T+T2, C, H, W, Z)
                vox_feats = self.post_unet(vox_feats)
                # (B, T2, C, H, W, Z)
                future_vox_feats = vox_feats[:, self.in_timesteps:]
                return [future_vox_feats_inter, future_vox_feats]
        else:
            # (B, T2, C, H, W, Z)
            future_vox_feats = future_vox_feats_inter
            return [future_vox_feats_inter, future_vox_feats]