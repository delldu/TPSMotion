"""Model Define."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022-2024, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 22日 星期四 03:54:39 CST
# ***
# ************************************************************************************/
#
import os
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Function
from torch.onnx import symbolic_helper

from typing import List

import todos
import pdb

# https://github.com/pytorch/pytorch/issues/107948
class MatrixInverse(Function):
    @staticmethod
    def forward(ctx, matrix) -> torch.Value:
        return torch.inverse(matrix)

    @staticmethod
    def symbolic(g: torch.Graph, matrix: torch.Value) -> torch.Value:
        return g.op("com.microsoft::Inverse", matrix)
matrix_inverse = MatrixInverse.apply


# The following comes from mmcv/ops/point_sample.py
def grid_sample(im, grid):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. 
    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.size()
    gn, gh, gw, _ = grid.size()
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    # if align_corners:
    #     x = ((x + 1) / 2) * (w - 1)
    #     y = ((y + 1) / 2) * (h - 1)
    # else:
    #     x = ((x + 1) * w - 1) / 2
    #     y = ((y + 1) * h - 1) / 2
    x = ((x + 1) / 2) * (w - 1)
    y = ((y + 1) / 2) * (h - 1)

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0.0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


class TPS:
    """TPS transformation for Eq(2) in the paper"""
    def __init__(self, bs: int, driving_kp, source_kp):
        # driving_kp.size() -- [1, 10, 5, 2]
        # source_kp.size() -- [1, 10, 5, 2]

        self.bs = bs  # 1
        self.gs = driving_kp.shape[1]  # 10
        N = driving_kp.shape[2]

        # Eq. (1), U(r) is a radial basis function, which represents the influence of each 
        # keypoint on the pixel at p
        # driving_kp[:, :, :, None].size() -- [1, 10, 5, 1, 2]
        # driving_kp[:, :, None, :].size() -- [1, 10, 1, 5, 2]
        K = torch.norm(driving_kp[:, :, :, None] - driving_kp[:, :, None, :], dim=4, p=2)
        # K.size() -- [1, 10, 5, 5]
        K = K**2
        K = K * torch.log(K + 1e-9)

        one = torch.ones(self.bs, self.gs, N, 1).to(driving_kp.device) # size() -- [1, 10, 5, 1]
        driving_kp_one = torch.cat([driving_kp, one], dim=3)  # driving_kp.size()--[1,10,5,2] ==> [1,10,5,3], xy-> xyz

        zero = torch.zeros(self.bs, self.gs, 3, 3).to(driving_kp.device) # size() -- [1, 10, 3, 3]
        P = torch.cat([driving_kp_one, zero], dim=2)  # [1, 10, 8, 3]
        L = torch.cat([K, driving_kp_one.permute(0, 1, 3, 2)], dim=2)
        L = torch.cat([L, P], dim=3)  # ==> # [1, 10, 8, 8]

        zero = torch.zeros(self.bs, self.gs, 3, 2).to(driving_kp.device) # [1, 10, 3, 2]
        Y = torch.cat([source_kp, zero], dim=2) # size() -- [1, 10, 8, 2]
        one = torch.eye(L.shape[2]).expand(L.shape).to(L.device)
        L = L + one * 0.01  # [1, 10, 8, 8]
        # param = torch.matmul(torch.inverse(L), Y) # size() -- [1, 10, 8, 2]
        param = torch.matmul(matrix_inverse(L), Y) # size() -- [1, 10, 8, 2]

        self.theta = param[:, :, N:, :].permute(0, 1, 3, 2)  # [1, 10, 2, 3]

        self.control_points = driving_kp  # [1, 10, 5, 2]
        self.control_params = param[:, :, :N, :]  # [1, 10, 5, 2]

    def warp_grid(self, grid):
        # grid.size() -- [1, 10, 64, 64, 2]
        theta = self.theta.to(grid.device)
        control_points = self.control_points.to(grid.device)
        control_params = self.control_params.to(grid.device)

        transformed = torch.matmul(theta[:, :, :, :2], grid.permute(0, 2, 1)) + theta[:, :, :, 2:]
        distances = (
            grid.view(grid.shape[0], 1, 1, -1, 2)
            - control_points.view(self.bs, control_points.shape[1], -1, 1, 2)
        )

        distances = distances**2
        result = distances.sum(-1)
        result = result * torch.log(result + 1e-9)

        result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
        transformed = transformed.permute(0, 1, 3, 2) + result

        return transformed  # [1, 10, 4096, 2]

    def transform_grid(self, frame):
        # frame.size() -- [1, 3, 64, 64]
        B, C, H, W = frame.size()
        grid = make_grid(H, W).unsqueeze(0).to(frame.device) # size() -- [1, 64, 64, 2]
        grid = grid.view(1, H * W, 2)  # [1, 4096, 2]

        # shape = [self.bs, self.gs, H, W, 2]
        grid = self.warp_grid(grid).view(self.bs, self.gs, H, W, 2)
        return grid  # [1, 10, 64, 64, 2]


def keypoint2gaussian(kp, H: int, W: int):
    """Transform a keypoint into gaussian like representation """
    # kp.size() -- [1, 50, 2]
    # H, W = 64, 64
    B, N, _ = kp.size()
    grid = make_grid(H, W).to(kp.device)  # [64, 64, 2]
    grid = grid.view(1, 1, H, W, 2)  # [1, 1, 64, 64, 2]
    grid = grid.repeat(B, N, 1, 1, 1)  # ==> [1, 50, 64, 64, 2]

    kp = kp.view(B, N, 1, 1, 2)
    mean = grid - kp
    sigma = 0.01
    out = torch.exp(-0.5 * (mean**2).sum(-1) / sigma)

    return out  # [1, 50, 64, 64]


def make_grid(H: int, W: int):
    """Create a meshgrid [-1,1] x [-1,1] of given (H, W). """
    # H = 64, W = 64
    x = torch.arange(W)
    y = torch.arange(H)

    x = 2.0 * (x / (W - 1.0)) - 1.0
    y = 2.0 * (y / (H - 1.0)) - 1.0

    yy = y.view(-1, 1).repeat(1, W)
    xx = x.view(1, -1).repeat(H, 1)

    # xx.size() -- torch.Size([64, 64])
    # xx.unsqueeze(2).size() ==> [64, 64, 1]
    meshed = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], dim=2)
    # tensor [meshed] size: [64, 64, 2], min: -1.0, max: 1.0, mean: 0.0

    return meshed


class ResBlock2d(nn.Module):
    """Res block, preserve spatial resolution. """
    def __init__(self, in_features, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features, out_channels=in_features, 
            kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features, out_channels=in_features, 
            kernel_size=kernel_size, padding=padding
        )
        self.norm1 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """Upsampling block for decoder. """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, 
            kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, recompute_scale_factor=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """Downsampling block for encoder. """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, 
            kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """Simple block, preserve spatial resolution. """
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, 
            kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class HourglassEncoder(nn.Module):
    def __init__(self, block_expansion=64, in_features=84, num_blocks=5, max_features=1024):
        super().__init__()
        down_blocks = []
        for i in range(num_blocks): # 5
            down_blocks.append(
                DownBlock2d(
                    in_features if i == 0 else min(max_features, block_expansion * (2**i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x) -> List[torch.Tensor]:
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class HourglassDecoder(nn.Module):
    def __init__(self, block_expansion=64, in_features=84, num_blocks=5, max_features=1024):
        super().__init__()
        up_blocks = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:
            in_filters = ((1 if i == num_blocks - 1 else 2)
                 * min(max_features, block_expansion * (2 ** (i + 1)))
            )
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2**i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)
        # ==> self.out_channels -- [1024, 2048, 1024, 512, 256, 148, 256]

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        out = x.pop()
        outs: List[torch.Tensor] = []
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        return outs


class Hourglass(nn.Module):
    """Hourglass architecture"""
    def __init__(self,
        block_expansion=64,
        in_features=84,
        num_blocks=5,
        max_features=1024,
    ):
        super().__init__()
        self.encoder = HourglassEncoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = HourglassDecoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels

    def forward(self, x) -> List[torch.Tensor]:
        #  tensor [x] size: [1, 84, 64, 64], min: -0.0, max: 0.998304, mean: 0.235856
        z = self.encoder(x)
        # z is list: len = 6
        #     tensor [item] size: [1, 84, 64, 64], min: -0.0, max: 0.998304, mean: 0.235856
        #     tensor [item] size: [1, 128, 32, 32], min: 0.0, max: 4.346566, mean: 0.212813
        #     tensor [item] size: [1, 256, 16, 16], min: 0.0, max: 3.665783, mean: 0.173401
        #     tensor [item] size: [1, 512, 8, 8], min: 0.0, max: 2.449036, mean: 0.166443
        #     tensor [item] size: [1, 1024, 4, 4], min: 0.0, max: 1.953367, mean: 0.092805
        #     tensor [item] size: [1, 1024, 2, 2], min: 0.0, max: 1.116109, mean: 0.031157        

        y = self.decoder(z)
        # y is list: len = 5
        #     tensor [item] size: [1, 2048, 4, 4], min: 0.0, max: 1.953367, mean: 0.057377
        #     tensor [item] size: [1, 1024, 8, 8], min: 0.0, max: 2.647899, mean: 0.139455
        #     tensor [item] size: [1, 512, 16, 16], min: 0.0, max: 4.250954, mean: 0.206135
        #     tensor [item] size: [1, 256, 32, 32], min: 0.0, max: 5.743032, mean: 0.240297
        #     tensor [item] size: [1, 148, 64, 64], min: -0.0, max: 4.772509, mean: 0.218149

        return y


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited down sampling, for better preservation of the input signal.
    """
    def __init__(self, channels=3, scale=0.25):
        super().__init__()
        sigma = (1 / scale - 1) / 2 # ==> sigma === 1.5
        kernel_size = 2 * round(sigma * 4) + 1 # ==> kernel_size === 13
        self.ka = kernel_size // 2 # ==> 6
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka # ==> 6

        kernel_size = [kernel_size, kernel_size] # [13, 13]
        sigma = [sigma, sigma] # [1.5, 1.5]
        
        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((grid - mean) ** 2) / (2 * std**2))

        kernel = kernel / torch.sum(kernel) # size() -- [13, 13]

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)) # size() -- [3, 1, 13, 13]

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        # tensor [input] size: [1, 3, 256, 256], min: 0.0, max: 1.0, mean: 0.629024
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale), 
            recompute_scale_factor=True)
        # tensor [out] size: [1, 3, 64, 64], min: 0.004092, max: 0.998304, mean: 0.623311
        return out


class KeyPointDetector(nn.Module):
    """
    Predict K*5 keypoints, here K == 10
    """
    def __init__(self, num_tps=10, model_path="models/drive_face.pth"):
        super().__init__()
        self.num_tps = num_tps
        self.fg_encoder = models.resnet18(weights=None)
        num_features = self.fg_encoder.fc.in_features  # 512
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)  # (512, 100)

        # first_kp for relative keypoints detection
        # self.register_buffer("first_kp", torch.zeros(1, self.num_tps * 5, 2))
        self.load_weights(model_path = model_path)

    def forward(self, image):
        return self.detect_source(image)

    # def detect_drive(self, image):
    #     # image.size() -- [1, 3, 256, 256]
    #     B, C, H, W = image.size()
    #     output_kp = self.detect_source(image)

    #     # relative keypoints case
    #     if self.first_kp.abs().max() < 1e-5:
    #         # ==> pdb.set_trace()
    #         print(f"Updating first key points ...")
    #         self.first_kp = output_kp[0:1, :, :]

    #     offset = output_kp - self.first_kp.repeat(B, 1, 1)

    #     return offset.tanh()/2.0

    def detect_source(self, image):
        # image.size() -- [1, 3, 256, 256]
        B, C, H, W = image.size()
        fg_kp = self.fg_encoder(image)  # size() -- [1, 100]
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2.0 - 1.0  # convert element from [0.0, 1.0] to [-1, 1.0]

        output_kp = fg_kp.view(B, self.num_tps * 5, -1)  # [1, 50, 2]
        return output_kp

    def load_weights(self, model_path="models/drive_face.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        self.load_state_dict(sd['kp_detector'])


class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks
           from K TPS trans_grid and an affine transformation.
    """
    def __init__(self,
        block_expansion=64,
        num_blocks=5,
        max_features=1024,
        num_tps=10,
        num_channels=3,
        scale_factor=0.25,
    ):
        super().__init__()

        self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.hourglass = Hourglass(
            block_expansion=block_expansion, # 64
            in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1), # why ???
            max_features=max_features, # 1024
            num_blocks=num_blocks, # 5
        )

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))
        # self.maps -- Conv2d(256, 11, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

        self.up_nums = int(math.log(1 / scale_factor, 2))  # 2
        self.occlusion_num = 4

        up = []
        channel = [hourglass_output_size[-1] // (2**i) for i in range(self.up_nums)]
        for i in range(self.up_nums):  # 2
            up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
        self.up = nn.ModuleList(up)

        channel = [hourglass_output_size[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
        for i in range(self.up_nums): # 2
            channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))
        # ==> channel -- [148, 256, 128, 64]
        # ==> hourglass_output_size -- [1024, 2048, 1024, 512, 256, 148, 256]

        occlusion = []
        for i in range(self.occlusion_num):  # 4
            occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
        self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps


    def create_heatmap(self, source_image, driving_kp, source_kp):
        B, C, H, W = source_image.size()
        gaussian_driving = keypoint2gaussian(driving_kp, H, W)
        gaussian_source = keypoint2gaussian(source_kp, H, W)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, H, W).to(heatmap.device)  # for Background ?
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_trans_grid(self, source_image, driving_kp, source_kp):
        # K TPS transformaions
        B, C, H, W = source_image.size()

        driving_kp = driving_kp.view(B, -1, 5, 2)
        source_kp = source_kp.view(B, -1, 5, 2)

        tps = TPS(B, driving_kp, source_kp)
        kp_grid = tps.transform_grid(source_image)

        id_grid = make_grid(H, W).to(source_kp.device) # for Background ?
        id_grid = id_grid.view(1, 1, H, W, 2)
        id_grid = id_grid.repeat(B, 1, 1, 1, 1)

        # id_grid.size() -- [1, 1, 64, 64, 2]
        # kp_grid.size() -- [1, 10, 64, 64, 2]
        grid = torch.cat([id_grid, kp_grid], dim=1)

        return grid  # [1, 11, 64, 64, 2]

    def create_deformed_source(self, source_image, trans_grid):
        # source_image.size() -- [1, 3, 64, 64]
        # trans_grid.size() -- [11, 64, 64, 2]
        B, C, H, W = source_image.size()
        source_repeat = source_image.repeat(self.num_tps + 1, 1, 1, 1) # size() -- [11, 3, 64, 64]
        # self.num_tps + 1 -- include backgroud information ?

        trans_grid = trans_grid.view((B * (self.num_tps + 1), H, W, -1)) # size() -- [11, 64, 64, 2])
        # onnx support
        # deformed_source = F.grid_sample(source_repeat, trans_grid, align_corners=True)
        deformed_source = grid_sample(source_repeat, trans_grid)

        return deformed_source.view(B, self.num_tps + 1, -1, H, W)  # [1, 11, 3, 64, 64]

    def forward(self, source_image, driving_kp, source_kp) -> List[torch.Tensor]:
        source_image = self.down(source_image) # AntiAliasInterpolation2d: size from 256x256 to 64x64

        B, C, H, W = source_image.size()
        # tensor [source_image] size: [1, 3, 64, 64], min: 0.004092, max: 0.998304, mean: 0.623311
        heatmap = self.create_heatmap(source_image, driving_kp, source_kp)
        # tensor [heatmap] size: [1, 51, 64, 64], min: -0.0, max: 0.0, mean: -0.0

        trans_grid = self.create_trans_grid(source_image, driving_kp, source_kp)
        # tensor [trans_grid] size: [1, 11, 64, 64, 2], min: -1.049396, max: 1.052836, mean: 0.000667

        deformed_source = self.create_deformed_source(source_image, trans_grid)
        # tensor [deformed_source] size: [1, 11, 3, 64, 64], min: 0.0, max: 0.998304, mean: 0.60036

        deformed_source = deformed_source.view(B, -1, H, W) # [1, 33, 64, 64]
        hourglass_input = torch.cat([heatmap, deformed_source], dim=1).view(B, -1, H, W)
        # tensor [input] size: [1, 84, 64, 64], min: -0.0, max: 0.998304, mean: 0.235856

        hourglass_output = self.hourglass(hourglass_input)  # List[torch.Tensor]
        # hourglass_output is list: len = 5
        #     tensor [item] size: [1, 2048, 4, 4], min: 0.0, max: 1.953367, mean: 0.057377
        #     tensor [item] size: [1, 1024, 8, 8], min: 0.0, max: 2.647899, mean: 0.139455
        #     tensor [item] size: [1, 512, 16, 16], min: 0.0, max: 4.250954, mean: 0.206135
        #     tensor [item] size: [1, 256, 32, 32], min: 0.0, max: 5.743032, mean: 0.240297
        #     tensor [item] size: [1, 148, 64, 64], min: -0.0, max: 4.772509, mean: 0.218149

        contribution_maps = self.maps(hourglass_output[-1])
        contribution_maps = F.softmax(contribution_maps, dim=1)
        # tensor [contribution_maps] size: [1, 11, 64, 64], min: 0.0, max: 0.999957, mean: 0.090909

        # Combine the K+1 trans_grid, Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2) # [1, 11, 1, 64, 64]
        trans_grid = trans_grid.permute(0, 1, 4, 2, 3) # ==> [1, 11, 2, 64, 64]
        optical_flow = (trans_grid * contribution_maps).sum(dim=1) # ==> [1, 2, 64, 64]
        optical_flow = optical_flow.permute(0, 2, 3, 1)
        # tensor [optical_flow] size: [1, 64, 64, 2], min: -1.000137, max: 1.000593, mean: 1.4e-05

        ################################################################
        dense_motion_output = [optical_flow]  # !!! Optical Flow !!!
        ################################################################

        # torch.jit.script not happy
        # for i in range(self.occlusion_num-self.up_nums):
        #     dense_motion_output.append(torch.sigmoid(self.occlusion[i](hourglass_output[self.up_nums-self.occlusion_num+i])))
        # hourglass_output = hourglass_output[-1]
        # for i in range(self.up_nums):
        #     prediction = self.up[i](prediction)
        #     dense_motion_output.append(torch.sigmoid(self.occlusion[i+self.occlusion_num-self.up_nums](hourglass_output)))
        for i, oc_block in enumerate(self.occlusion):  # 4
            if i < self.occlusion_num - self.up_nums:  # 2
                j = self.up_nums - self.occlusion_num + i
                # i --> [0, 1], j --> [-2, -1]
                t = oc_block(hourglass_output[j])
                dense_motion_output.append(torch.sigmoid(t))  # occlusion_map

        hourglass_output = hourglass_output[-1]
        for i, up_block in enumerate(self.up):  # 2
            hourglass_output = up_block(hourglass_output)
            for j, oc_block in enumerate(self.occlusion):  # 4
                if j == i + self.occlusion_num - self.up_nums:  # i->[0, 1], j -> [2, 3]
                    t = oc_block(hourglass_output)
                    dense_motion_output.append(torch.sigmoid(t))  # occlusion_map

        return dense_motion_output  # optical_flow + occlusion_map[...]


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels=3, block_expansion=64, max_features=512, num_down_blocks=3):
        super().__init__()

        self.num_down_blocks = num_down_blocks
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):  # 3
            in_features = min(max_features, block_expansion * (2**i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = [max_features, max_features, max_features // 2]
        out_features = [max_features // 2, max_features // 4, max_features // 8]
        for i in range(num_down_blocks):  # 3
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        resblock = []
        for i in range(num_down_blocks):  # 3
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock) # len(self.resblock) === 6

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

    def optical_flow_sample(self, input, optical_flow):
        # tensor [optical_flow] size: [1, 64, 64, 2], min: -1.000137, max: 1.000593, mean: 1.4e-05
        _, _, h, w = input.size()
        # _, h_old, w_old, _ = optical_flow..size()
        # if h_old != h or w_old != w:
        #     optical_flow = optical_flow.permute(0, 3, 1, 2)
        #     optical_flow = F.interpolate(optical_flow, size=(h, w), mode="bilinear", align_corners=True)
        #     optical_flow = optical_flow.permute(0, 2, 3, 1)
        # return F.grid_sample(input, optical_flow, align_corners=True)

        grid = optical_flow.permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=True)
        grid = grid.permute(0, 2, 3, 1)

        # onnx support            
        return grid_sample(input, grid)


    def forward(self, source_image, dense_motion: List[torch.Tensor]):
        out = self.first(source_image)
        encoder_map = [out]
        for i, block in enumerate(self.down_blocks):  # 3
            out = block(out)
            encoder_map.append(out)

        optical_flow = dense_motion[0]
        occlusion_map = dense_motion[1:]
        out = self.optical_flow_sample(out, optical_flow)
        out = out * occlusion_map[0] # mask 32x32 ?

        for i, up_block in enumerate(self.up_blocks):  # 3
            # torch.jit.script not happy
            # out = self.resblock[2*i](out)
            # out = self.resblock[2*i+1](out)
            for j, re_block in enumerate(self.resblock):
                if j == 2 * i or j == 2 * i + 1:
                    out = re_block(out)

            out = up_block(out)

            encode_i = encoder_map[-(i + 2)] # i --> [0, 1, 2], -(i + 2) --> [-2, -3, -4]
            encode_i = self.optical_flow_sample(encode_i, optical_flow)
            encode_i = encode_i * occlusion_map[i + 1] # mask

            if i < self.num_down_blocks - 1:
                out = torch.cat([out, encode_i], dim=1)

        deformed_source = self.optical_flow_sample(source_image, optical_flow)

        occlusion_last = occlusion_map[-1] # 256x256
        out = out * (1.0 - occlusion_last) + encode_i
        out = torch.sigmoid(self.final(out))
        out = out * (1.0 - occlusion_last) + deformed_source * occlusion_last

        return out


class ImageAnimation(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # self.keypoint_detector = KeyPointDetector()
        self.dense_motion = DenseMotionNetwork()
        self.generator = InpaintingNetwork()

        self.load_weights(model_path=model_path)
        self.eval()


    def load_weights(self, model_path="models/drive_face.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)

        # sd['kp_detector']['first_kp'] = torch.zeros(1, 50, 2) # add our init paramters
        self.generator.load_state_dict(sd['inpainting_network'])
        # self.keypoint_detector.load_state_dict(sd['kp_detector'])
        self.dense_motion.load_state_dict(sd['dense_motion_network'])

    # def forward(self, drive_tensor, source_tensor):
    def forward(self, source_kp, offset_kp, source_tensor):
        # tensor [source_kp] size: [1, 50, 2], min: -0.998097, max: 0.977059, mean: -0.015133
        # tensor [offset_kp] size: [1, 50, 2], min: -0.998392, max: 0.975835, mean: 0.013823
        # tensor [source_tensor] size: [1, 3, 256, 256], min: 0.0, max: 1.0, mean: 0.629024

        target_kp = source_kp + offset_kp

        dense_motion = self.dense_motion(source_tensor, target_kp, source_kp)
        # dense_motion is list: len = 5
        #     tensor [item] size: [1, 64, 64, 2], min: -1.000137, max: 1.000593, mean: 1.4e-05
        #     tensor [item] size: [1, 1, 32, 32], min: 0.0082, max: 1.0, mean: 0.442793
        #     tensor [item] size: [1, 1, 64, 64], min: 0.090537, max: 0.99181, mean: 0.736101
        #     tensor [item] size: [1, 1, 128, 128], min: 0.02473, max: 0.999199, mean: 0.795737
        #     tensor [item] size: [1, 1, 256, 256], min: 0.348033, max: 0.999995, mean: 0.997214

        out = self.generator(source_tensor, dense_motion)
        return out.clamp(0.0, 1.0)
