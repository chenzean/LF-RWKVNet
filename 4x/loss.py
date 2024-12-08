import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from imageio import imsave
import math
from einops import rearrange

def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    Charbonnier_loss = torch.sum(error) / torch.numel(error)
    return Charbonnier_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, sr, hr):
        rec_loss = L1_Charbonnier_loss(sr, hr)
        return rec_loss


class DetailLoss(nn.Module):
    def __init__(self):
        super(DetailLoss, self).__init__()

    def forward(self, sr, hr):    # [B*U*V, 1, H, W]
        sr_grad1 = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        hr_grad1 = hr[:, :, 1:, :] - hr[:, :, :-1, :]
# -------------------------------------------------------------------------------------------------------------
        sr_grad2 = sr[:, :, :, 1:] - sr[:, :, :, :-1]
        hr_grad2 = hr[:, :, :, 1:] - hr[:, :, :, :-1]

        detail_4d_loss1 = L1_Charbonnier_loss(sr_grad1, hr_grad1)
        detail_4d_loss2 = L1_Charbonnier_loss(sr_grad2, hr_grad2)
        detail_4d_loss = (detail_4d_loss1 + detail_4d_loss2) / 2.
        return detail_4d_loss




class alignLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.an = args.angRes
        self.an2 = args.angRes * args.angRes


    def forward(self, feat_gt, aligned_1):
        '''
        depth map: (b an2) c h w
        '''
        flow_loss = L1_Charbonnier_loss(aligned_1, feat_gt)
        return flow_loss


class Spectral_Loss(nn.Module):
    def __init__(self, Window_size=3):
        super(Spectral_Loss, self).__init__()
        # self.mse_loss = nn.MSELoss(reduction='mean')
        self.window_size = Window_size

        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, sr, hr):
        sr_fft2 = torch.fft.fft2(sr)
        # sr_fft2_real = sr_fft2.real
        # sr_fft2_imag  = sr_fft2.imag
        hr_fft2 = torch.fft.fft2(hr)
        # sr_fft2_real = hr_fft2.real
        # sr_fft2_imag  = hr_fft2.imag

        # Local spectrum loss
        spectrum_input = torch.abs(sr_fft2.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1))
        spectrum_target = torch.abs(hr_fft2.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1))
        local_spectrum_loss = L1_Charbonnier_loss(spectrum_input, spectrum_target)

        # Local phase loss
        phase_input = torch.angle(sr_fft2.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1))
        phase_target = torch.angle(hr_fft2.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1))
        local_phase_loss = L1_Charbonnier_loss(phase_input, phase_target)

        # Joint loss
        joint_loss = L1_Charbonnier_loss(spectrum_input * phase_input, spectrum_target * phase_target)

        return self.alpha * local_spectrum_loss + self.beta * local_phase_loss + (1 - self.alpha- self.beta) * joint_loss

class EPILoss(nn.Module):
    def __init__(self):
        super(EPILoss, self).__init__()

    def forward(self, sr, hr, srb, sran2, srh, srw):   # [B*U*V, 1, H, W]
        sran = int(np.sqrt(sran2))
        sr = sr.view([srb, sran, sran, srh, srw]).unsqueeze(dim=1)  # [B, U, V, H, W] to [B, 1, U, V, H, W]
        hr = hr.view([srb, sran, sran, srh, srw]).unsqueeze(dim=1)  # [B, U, V, H, W] to [B, 1, U, V, H, W]

        sr_epi1 = sr.permute(0, 2, 4, 3, 5, 1)    # [B, U, H, V, W, C]
        hr_epi1 = hr.permute(0, 2, 4, 3, 5, 1)    # [B, U, H, V, W, C]

        sr_epi2 = sr.permute(0, 3, 5, 2, 4, 1)    # [B, V, W, U, H, C]
        hr_epi2 = hr.permute(0, 3, 5, 2, 4, 1)    # [B, V, W, U, H, C]

        sr_epi1_grad = sr_epi1[:, :, :, 1:, :, :] - sr_epi1[:, :, :, :-1, :, :]
        hr_epi1_grad = hr_epi1[:, :, :, 1:, :, :] - hr_epi1[:, :, :, :-1, :, :]

        sr_epi2_grad = sr_epi2[:, :, :, :, 1:, :] - sr_epi2[:, :, :, :, :-1, :]
        hr_epi2_grad = hr_epi2[:, :, :, :, 1:, :] - hr_epi2[:, :, :, :, :-1, :]

        epi_loss1 = L1_Charbonnier_loss(sr_epi1_grad, hr_epi1_grad)
        epi_loss2 = L1_Charbonnier_loss(sr_epi2_grad, hr_epi2_grad)

        epi_loss = (epi_loss1 + epi_loss2) / 2.0
        return epi_loss

# class depthLoss(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args =args
#         self.an = args.angRes
#         self.an2 =args.angRes * args.angRes

    # def forward(self, depth_map, HR, ref):
    #     '''
    #     depth map: (b an2) c h w
    #     '''

        # b, c, H, W = HR.shape
        # h_ = H//self.an
        # w_ = W//self.an
        #
        # HR_stack = rearrange(HR, 'b c (an1 h) (an2 w)->(b an1 an2) c h w', an1=self.an, an2=self.an, h=h_, w=w_)
        #
        # # center_view = []
        # # for b in range(b):
        # #     center = HR_stack[b, 12, :, :, :]
        # #     center = center[None, None, :, :, :]
        # #     center_view.append(center)
        # #
        # # center_view = torch.cat(center_view, dim=0)
        #
        # warp_loss_total = 0
        # j = 0
        # for i in range(self.an2 * b):
        #     if j+b<self.an2 *b:
        #         dis = depth_map[j:j+b,:,:,:]
        #         # dis = dis[None,:,:,:]
        #         j = j + b
        #         temp_hr = HR_stack[i,:,:,:]
        #         warped_ima = self.flow_warp(ref, dis.permute(0, 2, 3, 1), 'bilinear' )
        #
        #         warping_loss = L1_Charbonnier_loss(warped_ima, temp_hr)
        #
        #         warp_loss_total = warping_loss + warp_loss_total

        # return warp_loss_total/(self.an2 * b)

    def flow_warp(self,
                  x,
                  flow,
                  interpolation='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        """
        if x.size()[-2:] != flow.size()[1:3]:
            flow = F.interpolate(flow, size=x.size()[-2:], mode='bilinear')
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                             f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
        grid.requires_grad = False

        grid_flow = grid + flow  # grid_flow的维度
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        output = F.grid_sample(x,  # x的维度[4, 3, 3, 3]
                               grid_flow,  # grid_flow的维度[1,3,3,2]
                               mode=interpolation,
                               padding_mode=padding_mode,
                               align_corners=align_corners)
        return output



# def warping(disp, ind_source, ind_target, img_source, an):
#     '''warping one source image/map to the target'''
#     # an angular number
#     # disparity: int or [N,h,w]
#     # ind_souce 要移动图像的索引
#     # ind_target 索引值
#     # img_source [N,c,h,w]   需要移动的图像
#
#     # ==> out [N,c,h,w]
#     # print('img source ', img_source.shape)
#
#     N, c, h, w = img_source.shape
#     disp = disp.type_as(img_source)
#     # ind_source = ind_source.type_as(disp)
#     # ind_target = ind_target.type_as(disp)
#     # print(img_source.shape)
#     # coordinate for source and target
#     # ind_souce = torch.tensor([0,an-1,an2-an,an2-1])[ind_source]
#     ind_h_source = math.floor(ind_source / an)
#     ind_w_source = ind_source % an
#
#     ind_h_target = math.floor(ind_target / an)
#     ind_w_target = ind_target % an
#
#     # generate grid
#     XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
#     YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)
#
#     grid_w = XX + disp * (ind_w_target - ind_w_source)
#     grid_h = YY + disp * (ind_h_target - ind_h_source)
#
#     grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0
#     grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0
#
#     grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)  # [N,h,w,2]
#
#     # img_source = torch.unsqueeze(img_source, 1)
#     # print(img_source.shape)
#     # print(grid.shape)
#     # print(tt)
#     img_target = F.grid_sample(img_source, grid, padding_mode='border', align_corners=False)  # [N,3,h,w]
#     # img_target = torch.squeeze(img_target, 1)  # [N,h,w]
#     return img_target
def img_grads(I):
    I_dy = I[:, :, 1:, :] - I[:, :, :-1, :]
    I_dx = I[:, :, :, 1:] - I[:, :, :, :-1]
    return I_dx, I_dy

class Edge_Aware_Smoothness_Loss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.an = args.angRes


    def forward(self, dis, GT, edge_constant=150):
        ban2, c, h, w = GT.shape
        batch = ban2//(self.an * self.an)
        h_ = h//self.an
        w_ = w//self.an
        GT = rearrange(GT, 'b c (an1 h) (an2 w)->(b an1 an2) c h w',h=h_,w=w_,an1=self.an,an2=self.an)
        img_gx_y, img_gy_y = img_grads(GT[:, 0:1])
        # img_gx_r, img_gy_r = img_grads(GT[:, 0:1])
        # img_gx_g, img_gy_g = img_grads(GT[:, 1:2])
        # img_gx_b, img_gy_b = img_grads(GT[:, 2:3])

        weight_x = torch.exp(-edge_constant * torch.abs(img_gx_y))
        weight_y = torch.exp(-edge_constant * torch.abs(img_gy_y))

        disp_gx, disp_gy = img_grads(dis)

        loss = (torch.mean(weight_x * torch.abs(disp_gx)) + torch.mean(weight_y * torch.abs(disp_gy)))/2.

        return loss




def get_loss_dict(args):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss()
        loss['detail_loss'] = DetailLoss()
        loss['Spectral_loss'] = Spectral_Loss()

    if (abs(args.epi_w - 0) > 1e-8):
        loss['epi_loss'] = EPILoss()
    return loss


