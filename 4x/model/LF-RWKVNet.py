import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.nn import init

from loss import ReconstructionLoss, DetailLoss
from model.fusion_block import *

# from model.SAS_conv import SAS_Conv2D
from model.Restore_RWKV import EPI_Block


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

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        self.args = args

        self.channels = args.channels
        self.angRes = args.angRes
        self.factor = args.scale_factor
        self.patch_size = args.patch_size
        self.device = args.device

        self.conv_init = nn.Sequential(
            nn.Conv2d(1, self.channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResASPP(self.channels)
        )

        self.pixel_down1 = nn.PixelUnshuffle(2)
        self.pixel_down2 = nn.PixelUnshuffle(4)

        # forward中没用到,但是训练权重中有这个内容，若计算参数量需要将其注释
        self.pixel_up2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2 * 2, kernel_size=1),
            nn.PixelShuffle(2),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
        )

        self.pixel_up1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2 * 2, kernel_size=1),
            nn.PixelShuffle(2),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
        )

        # 4x 分支
        self.FB1 = FusionBlock_new3_new_method(args, self.channels)
        self.conv_up4_1 = nn.Conv2d(self.channels * 2 * 2 - self.channels, self.channels  * 2 * 2, kernel_size=1)
        self.FB2 = FusionBlock_new3_new_method(args, self.channels)
        self.conv_up4_2 = nn.Conv2d(self.channels * 2 * 2 - self.channels, self.channels * 2 * 2, kernel_size=1)
        self.FB3 = FusionBlock_new3_new_method(args, self.channels )
        self.conv_up4_3 = nn.Conv2d(self.channels * 2 * 2 - self.channels, self.channels, kernel_size=1)
        self.FB4 = FusionBlock_new3_new_method(args, self.channels )

        self.ca_4x = nn.Sequential(
            SELayer(self.channels * 4),
            nn.Conv2d(self.channels * 4, self.channels, kernel_size=1),
        )
        self.RB_4x = RB(self.channels)

        # 8x 分支
        self.FB7 = FusionBlock_new3_new_method(args,  self.channels)

        self.ca_8x = nn.Sequential(
            SELayer(self.channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
        )
        self.RB_8x = RB(self.channels)


        # 4x 分支
        self.VSSBlock1 = EPI_Block(args, self.args.channels)
        self.VSSBlock2 = EPI_Block(args, self.args.channels)
        self.VSSBlock3 = EPI_Block(args, self.args.channels)
        self.VSSBlock4 = EPI_Block(args, self.args.channels)

        # 8x 分支
        self.VSSBlock7 = EPI_Block(args, self.args.channels)

        self.conv_4 = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1)

    def forward(self, lr, ref, data_info=None):
        '''
               Args:
               shape of lr images is [Bath_size, channels, (angRes h), (angRes w)]     for example: [1, 1, 160, 160]   (SAI array)
               shape of ref images is [Bath_size, channels, H, W]                      for example: [1, 1, 128, 128]
               '''
        b, c, H, W = lr.shape
        h_ = H // self.angRes
        w_ = W // self.angRes

        lr_up8_low_freq = interpolate(lr, scale_factor=self.factor, mode='bicubic', angRes=self.angRes)
        lr_up2 = interpolate(lr, self.angRes, scale_factor=2, mode='bicubic')
        lr_up2_stack = rearrange(lr_up2, 'b c (an1 h) (an2 w)->(b an1 an2) c h w', b=b, an1=self.angRes,
                                 an2=self.angRes,
                                 h=h_ * 2,
                                 w=w_ * 2)

        lr_up2_stack_fea = self.conv_init(lr_up2_stack)
        ref_fea = self.conv_init(ref)

        ref_fea_down_1 = self.pixel_down1(ref_fea)

        # 4x 分支
        # ==============================================================================================
        ref_feature1, buffer = channels_split(ref_fea_down_1, self.channels)
        migration_fea_1 = self.FB1(lr_up2_stack_fea, ref_feature1)  # [25, channels, 48, 48]
        fea_4x_1 = self.VSSBlock1(migration_fea_1)  # [25, channels, 48, 48]
        fea_4x_up = self.conv_up4_1(buffer)

        ref_feature1, buffer = channels_split(fea_4x_up, self.channels)
        migration_fea_2 = self.FB2(fea_4x_1, ref_feature1)  # [25, channels, 48, 48]
        fea_4x_2 = self.VSSBlock2(migration_fea_2)  # [25, channels, 48, 48]
        fea_4x_up = self.conv_up4_2(buffer)

        ref_feature1, buffer = channels_split(fea_4x_up, self.channels)
        migration_fea_2 = self.FB3(fea_4x_2, ref_feature1)  # [25, channels, 48, 48]
        fea_4x_3 = self.VSSBlock3(migration_fea_2)  # [25, channels, 48, 48]
        fea_4x_up = self.conv_up4_3(buffer)

        migration_fea_2 = self.FB4(fea_4x_3, fea_4x_up)  # [25, channels, 48, 48]
        fea_4x_4 = self.VSSBlock4(migration_fea_2)  # [25, channels, 48, 48]

        fea = torch.cat([fea_4x_1, fea_4x_2, fea_4x_3, fea_4x_4], dim=1)

        fea_out_4x = self.ca_4x(fea)
        fea_out_4x = self.RB_4x(fea_out_4x)

        out2 = self.conv_4(fea_out_4x)
        out2 = rearrange(
            out2, '(b an1 an2) c h w -> b c (an1 h) (an2 w)', an1=self.angRes, an2=self.angRes) + lr_up2

        fea_up_1 = self.pixel_up1(fea_out_4x)  # [25, channels, 96, 96]

        # 8x 分支
        # ==============================================================================================
        fusion_fea_1 = self.FB7(fea_up_1, ref_fea)  # [25, channels, 96, 96]
        fea_7 = self.VSSBlock7(fusion_fea_1)  # [25, channels, 96, 96]

        fea_out_8x = self.ca_8x(fea_7)
        fea_out_8x = self.RB_8x(fea_out_8x)

        out = self.conv_out(fea_out_8x)

        output = rearrange(
            out, '(b an1 an2) c h w -> b c (an1 h) (an2 w)', an1=self.angRes, an2=self.angRes) + lr_up8_low_freq

        return output, out2

def interpolate(x, angRes, scale_factor, mode):
    '''
    up-sampling: Bicubic Interpolation
    input:[B, 1, H, W]
    output:[B, 1, H * scale_factor, W * scale_factor]
    '''
    [B, _, H, W] = x.size()
    h = H // angRes         
    w = W // angRes     
    x_upscale = x.view(B, 1, angRes, h, angRes, w) 
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)  
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)  
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)           
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor, W * scale_factor)
    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.Rec_Loss = ReconstructionLoss()
        self.Detail_Loss = DetailLoss()

    def forward(self, output, out4, label, hr_down2, data_info=None):
        rec_loss8 = self.Rec_Loss(output, label)
        detail_loss8 = self.Detail_Loss(output, label)
        loss8 = rec_loss8 + detail_loss8

        rec_loss4 = self.Rec_Loss(out4, hr_down2)
        detail_loss4 = self.Detail_Loss(out4, hr_down2)
        loss4 = rec_loss4 + detail_loss4


        return loss8 + loss4



class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel,channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.ReLU(inplace=True))
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1

def weights_init(m):
    pass

def channels_split(ref_feature, channels):
    """
    将 ref_feature 按照给定的 channels 数量进行分割，返回第一部分和剩余部分
    """
    ref_feature1 = ref_feature[:, :channels, :, :]  # 前 channels 个通道
    ref_feature2 = ref_feature[:, channels:, :, :]  # 剩余的通道

    return ref_feature1, ref_feature2


if __name__ == "__main__":
    from thop import profile
    from config import args
    net = get_model(args).cuda()
    input_lr = torch.randn(1,1,60,60).cuda()
    input_ref = torch.randn(1, 1, 96, 96).cuda()
    flops, params = profile(net, inputs=(input_lr,input_ref))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
