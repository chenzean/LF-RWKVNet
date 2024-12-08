# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F 
from einops import rearrange, repeat
from model.SAS_conv import SAS_Conv2D
from model.fusion_block import Attention
from torchvision.ops import DeformConv2d
T_MAX = 114*114


from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["./model/cuda/wkv_op.cpp", "./model/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平局池化操作
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 全连接
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 全连接
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入特征图 x 的大小，b 表示批次的大小，c 表示输入特征图的通道数
        y = self.avg_pool(x).view(b, c)
        # 输入特征图 x 经过平均池化层 self.avg_pool 处理，然后使用 view 方法将其形状变为 (b, c)。这一步是为了将特征图转换为向量的形式
        y = self.fc(y).view(b, c, 1, 1)
        # 特征向量 y 经过全连接层 self.fc 处理，然后使用 view 方法将其形状变为 (b, c, 1, 1)。这一步是为了将特征向量转换为与输入特征图相同的形状
        return x * y.expand_as(x)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

"""Dynamic Snake Convolution Module"""
class DSConv_pro(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str = "cuda:0",
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Sequential(
            SpatialAttention(),
            nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1),
        )
        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output

def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str = "cuda:1",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map

def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature

def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled

# ===============================================================================

class snake_conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_x = DSConv_pro(dim, dim,5,morph=0)
        self.conv_y = DSConv_pro(dim, dim,5,morph=1)
        self.conv_0 = nn.Conv2d(dim, dim,(3,5), (1, 1), (1,2))
        self.conv3x3 = nn.Conv2d(dim * 3, dim, 1)
        self.ca = SELayer(dim * 3, dim)

    def forward(self, x):
        fea_x = self.conv_x(x)
        fea_y = self.conv_y(x)
        fea_0 = self.conv_0(x)

        fea_all = torch.cat([fea_x, fea_0, fea_y], dim=1)
        fea_all = self.ca(fea_all)
        fea = self.conv3x3(fea_all)

        return fea

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 5, padding=2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sigmoid(self.conv1(x2))
        return sattn * x

class VRWKV_EPIMix(nn.Module):
    def __init__(self, n_embd, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.recurrence = 2

        self.omni_shift = snake_conv(n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()

        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        self.omni_shift = snake_conv(n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x

class LF_EPI_Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_EPIMix(n_embd, init_mode,
                                    key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate,
                                   init_mode, key_norm=key_norm)


        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        resolution = (h, w)

        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x * self.gamma1 + self.att(self.ln1(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # x = self.dwconv2(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x * self.gamma2 + self.ffn(self.ln2(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class RB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        return fea + x

class EPI_Block(nn.Module):
    def __init__(self, args, n_embd, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()

        self.args = args

        self.conv_seq1 = nn.Sequential(
            RB(n_embd),
        )
        self.conv_seq2 = nn.Sequential(
            RB(n_embd),
        )

        self.EPI_RWKV1 = LF_EPI_Block(n_embd, hidden_rate)
        self.EPI_RWKV2 = LF_EPI_Block(n_embd, hidden_rate)

    def forward(self, x):

        b, c, h, w = x.shape

        fea = rearrange(x, '(b an1 an2) c h w -> (b an2 w) c an1 h', an1=self.args.angRes, an2=self.args.angRes, h=h, w=w)
        fea = self.EPI_RWKV1(fea)
        fea = rearrange(fea, '(b an2 w) c an1 h -> (b an1 an2) c h w', an1=self.args.angRes, an2=self.args.angRes, h=h, w=w)
        fea = self.conv_seq1(fea)

        fea = rearrange(fea, '(b an1 an2) c h w ->(b an1 h) c an2 w ', an1=self.args.angRes, an2=self.args.angRes, h=h, w=w)
        fea = self.EPI_RWKV2(fea)
        fea = rearrange(fea, '(b an1 h) c an2 w ->(b an1 an2) c h w ', an1=self.args.angRes, an2=self.args.angRes, h=h, w=w)
        fea = self.conv_seq2(fea)

        return fea
