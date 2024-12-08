# -*- coding: utf-8 -*-
# @Time    : 2024/7/18 10:48
# @Author  : Chen Zean
# @Site    : 
# @File    : fusion_block.py
# @Software: PyCharm

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numbers
from torchvision.ops import DeformConv2d
import math
import matplotlib.pyplot as plt

class SELayer(nn.Module):
    # SE注意力机制
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

class FusionBlock_new3_new_method(nn.Module):
    def __init__(self, args, ref_channels, last_flag=True):
        super(FusionBlock_new3_new_method, self).__init__()

        self.args = args
        self.ref_channels = ref_channels
        self.channels = args.channels

        # 动态创建 cross attention block 和 conv_1x1_up 层
        self.channels_cross_attention = TransformerBlock(self.channels, self.ref_channels, 4, 2.66, False)

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, 1),
        )

    def forward(self, lr_feature, ref_feature):
        b, c, h, w = lr_feature.shape
        u = self.args.angRes
        v = self.args.angRes

        if lr_feature.shape[0] != ref_feature.shape[0]:
            repeat_num = lr_feature.shape[0] // ref_feature.shape[0]
            # 运行速度会慢一些
            # ref_feature = ref_feature.repeat(repeat_num, 1, 1, 1)
            # 速度会快一些，原因：这不会实际分配新的内存，而是会创建一个可广播的视图
            ref_feature = ref_feature.expand(repeat_num, -1, -1, -1)

        lr_ref_feature = self.channels_cross_attention(lr_feature, ref_feature)

        fea = torch.cat([lr_feature, lr_ref_feature], dim=1)

        fea_out = self.conv(fea)

        return fea_out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class TransformerBlock(nn.Module):
    def __init__(self, dim, split_channels, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_ref = LayerNorm(split_channels, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, split_channels, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, lr, ref):
        lr_ln = self.norm1(lr)
        ref_ln = self.norm_ref(ref)
        lr = lr + self.attn(lr_ln, ref_ln)
        out = lr + self.ffn(self.norm2(lr))

        return out

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, ref_dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(ref_dim, ref_dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(ref_dim * 2, ref_dim * 2, kernel_size=3, stride=1, padding=1, groups=ref_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.relu = nn.ReLU(inplace=True)
        # self.w = nn.Parameter(torch.ones(2))


    def forward(self, lr, ref):
        b, c, h, w = lr.shape

        q = self.q_dwconv(self.q(lr))
        kv = self.kv_dwconv(self.kv(ref))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn1 = attn.softmax(dim=-1)
        # attn2 = self.relu(attn)**2

        out = (attn1 @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
