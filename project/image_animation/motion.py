"""Model Define."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 22日 星期四 03:54:39 CST
# ***
# ************************************************************************************/
#
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from typing import List
import math
import pdb


class TPS:
    """
    TPS transformation for Eq(2) in the paper
    """

    def __init__(self, bs: int, kp_1, kp_2):
        # kp_1 -- [1, 10, 5, 2]
        # kp_2 -- [1, 10, 5, 2]

        self.bs = bs  # 1
        self.gs = kp_1.shape[1]  # 10
        N = kp_1.shape[2]

        K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
        # K.size() -- [1, 10, 5, 5]
        K = K ** 2
        K = K * torch.log(K + 1e-9)

        one = torch.ones(self.bs, self.gs, N, 1).to(kp_1.device)
        # one.size() -- [1, 10, 5, 1]
        kp_1p = torch.cat([kp_1, one], dim=3)  # kp_1.size()--[1,10,5,2] ==> [1,10,5,3]

        zero = torch.zeros(self.bs, self.gs, 3, 3).to(kp_1.device)
        P = torch.cat([kp_1p, zero], dim=2)  # [1, 10, 8, 3]
        L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], dim=2)
        L = torch.cat([L, P], dim=3)  # ==> # [1, 10, 8, 8]

        zero = torch.zeros(self.bs, self.gs, 3, 2).to(kp_1.device)
        Y = torch.cat([kp_2, zero], dim=2)
        one = torch.eye(L.shape[2]).expand(L.shape).to(kp_1.device) * 0.01
        L = L + one  # [1, 10, 8, 8]

        param = torch.matmul(torch.inverse(L), Y)
        self.theta = param[:, :, N:, :].permute(0, 1, 3, 2)  # [1, 10, 2, 3]

        self.control_points = kp_1  # [1, 10, 5, 2]
        self.control_params = param[:, :, :N, :]  # [1, 10, 5, 2]

    def transform_frame(self, frame):
        # frame.size() -- [1, 3, 64, 64]
        B, C, H, W = frame.shape
        grid = make_coordinate_grid(H, W).unsqueeze(0).to(frame.device)
        # grid.size() -- [1, 64, 64, 2]
        grid = grid.view(1, H * W, 2)  # [1, 4096, 2]

        # shape = [self.bs, self.gs, H, W, 2]
        grid = self.warp_coordinates(grid).view(self.bs, self.gs, H, W, 2)
        return grid  # [1, 10, 64, 64, 2]

    def warp_coordinates(self, coordinates):
        theta = self.theta.to(coordinates.device)
        control_points = self.control_points.to(coordinates.device)
        control_params = self.control_params.to(coordinates.device)

        transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]
        distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(
            self.bs, control_points.shape[1], -1, 1, 2
        )

        distances = distances ** 2
        result = distances.sum(-1)
        result = result * torch.log(result + 1e-9)
        result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
        transformed = transformed.permute(0, 1, 3, 2) + result

        return transformed  # [1, 10, 4096, 2]


def kp2gaussian(kp, H: int, W: int):
    """
    Transform a keypoint into gaussian like representation
    """
    # kp.size() -- [1, 50, 2]
    # H, W = 64, 64

    B, N, _ = kp.shape
    coordinate_grid = make_coordinate_grid(H, W).to(kp.device)  # [64, 64, 2]
    coordinate_grid = coordinate_grid.view(1, 1, H, W, 2)  # [1, 1, 64, 64, 2]
    coordinate_grid = coordinate_grid.repeat(B, N, 1, 1, 1)  # ==> [1, 50, 64, 64, 2]

    kp = kp.view(B, N, 1, 1, 2)
    mean_sub = coordinate_grid - kp
    kp_variance = 0.01
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out  # [1, 50, 64, 64]


def make_coordinate_grid(H: int, W: int):
    """
    Create a meshgrid [-1,1] x [-1,1] of given (H, W).
    """
    x = torch.arange(W)
    y = torch.arange(H)

    x = 2.0 * (x / (W - 1.0)) - 1.0
    y = 2.0 * (y / (H - 1.0)) - 1.0

    yy = y.view(-1, 1).repeat(1, W)
    xx = x.view(1, -1).repeat(H, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], dim=2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding
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
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, recompute_scale_factor=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups
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
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        # print('encoder:' ,outs[-1].shape)
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
            # print('encoder:' ,outs[-1].shape)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)
        # self.out_filters = block_expansion + in_features

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
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels
        # self.out_filters = self.decoder.out_filters

    def forward(self, x) -> List[torch.Tensor]:
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale=0.25):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        # if self.scale == 1.0: # self.scale -- 0.25
        #     return input
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale), recompute_scale_factor=True)

        return out


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints
    """

    def __init__(self, num_tps=10):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps  # K -- 10
        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features  # 512
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)  # (512, 100)

    def forward(self, image):
        # image.size() -- [1, 3, 256, 256]
        fg_kp = self.fg_encoder(image)  # [1, 100]
        B, C = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2.0 - 1.0  # fg_kp.size() -- [1, 100], [-1, 1.0]

        return fg_kp.view(B, self.num_tps * 5, -1)  # [1, 50, 2]


class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks
           from K TPS transformations and an affine transformation.
    """

    def __init__(
        self, block_expansion=64, num_blocks=5, max_features=1024, num_tps=10, num_channels=3, scale_factor=0.25
    ):
        super(DenseMotionNetwork, self).__init__()

        self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor

        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
            max_features=max_features,
            num_blocks=num_blocks,
        )

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))

        up = []
        self.up_nums = int(math.log(1 / scale_factor, 2))  # 2
        self.occlusion_num = 4

        channel = [hourglass_output_size[-1] // (2 ** i) for i in range(self.up_nums)]
        for i in range(self.up_nums):  # 2
            up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
        self.up = nn.ModuleList(up)

        channel = [hourglass_output_size[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
        for i in range(self.up_nums):
            channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))

        occlusion = []
        for i in range(self.occlusion_num):  # 4
            occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
        self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps

    def create_heatmap(self, source_image, kp_driving, kp_source):
        B, C, H, W = source_image.shape
        gaussian_driving = kp2gaussian(kp_driving, H, W)
        gaussian_source = kp2gaussian(kp_source, H, W)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, H, W).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source):
        # K TPS transformaions
        B, C, H, W = source_image.shape

        kp_driving = kp_driving.view(B, -1, 5, 2)
        kp_source = kp_source.view(B, -1, 5, 2)
        trans = TPS(B, kp_driving, kp_source)
        driving_to_source = trans.transform_frame(source_image)

        identity_grid = make_coordinate_grid(H, W).to(kp_source.device)
        identity_grid = identity_grid.view(1, 1, H, W, 2)
        identity_grid = identity_grid.repeat(B, 1, 1, 1, 1)

        # identity_grid.size() -- [1, 1, 64, 64, 2]
        # driving_to_source.size() -- [1, 10, 64, 64, 2]
        transformations = torch.cat([identity_grid, driving_to_source], dim=1)

        return transformations  # [1, 11, 64, 64, 2]

    def create_deformed_source(self, source_image, transformations):
        # source_image.size() -- [1, 3, 64, 64]
        # transformations.size() -- [11, 64, 64, 2]
        B, C, H, W = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(B * (self.num_tps + 1), -1, H, W)
        transformations = transformations.view((B * (self.num_tps + 1), H, W, -1))
        deformed_source = F.grid_sample(source_repeat, transformations, align_corners=True)
        return deformed_source.view(B, self.num_tps + 1, -1, H, W)  # # [1, 11, 3, 64, 64]

    def forward(self, source_image, kp_driving, kp_source) -> List[torch.Tensor]:
        source_image = self.down(source_image)
        B, C, H, W = source_image.shape

        heatmap = self.create_heatmap(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source(source_image, transformations)
        deformed_source = deformed_source.view(B, -1, H, W)
        input = torch.cat([heatmap, deformed_source], dim=1).view(B, -1, H, W)

        prediction = self.hourglass(input)  # List[torch.Tensor]

        contribution_maps = self.maps(prediction[-1])
        contribution_maps = F.softmax(contribution_maps, dim=1)

        # Combine the K+1 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        dense_motion_output = [deformation]  # Optical Flow

        for i, oc_block in enumerate(self.occlusion):  # 4
            if i < self.occlusion_num - self.up_nums:  # 2
                j = self.up_nums - self.occlusion_num + i
                # i --> [0, 1], j --> [-2, -1]
                t = oc_block(prediction[j])
                dense_motion_output.append(torch.sigmoid(t))  # occlusion_map

        prediction = prediction[-1]
        for i, up_block in enumerate(self.up):  # 2
            prediction = up_block(prediction)
            for j, oc_block in enumerate(self.occlusion):  # 4
                if j == i + self.occlusion_num - self.up_nums:  # i->[0, 1], j -> [2, 3]
                    t = oc_block(prediction)
                    dense_motion_output.append(torch.sigmoid(t))  # occlusion_map

        return dense_motion_output  # deformation, occlusion_map[...]


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """

    def __init__(self, num_channels=3, block_expansion=64, max_features=512, num_down_blocks=3):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):  # 3
            in_features = min(max_features, block_expansion * (2 ** i))
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
        self.resblock = nn.ModuleList(resblock)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        # self.num_channels = num_channels

    def deform_input(self, input, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = input.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode="bilinear", align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(input, deformation, align_corners=True)

    def occlude_input(self, input, occlusion_map):
        return input * occlusion_map

    def forward(self, source_image, dense_motion: List[torch.Tensor]):
        out = self.first(source_image)
        encoder_map = [out]
        for i, block in enumerate(self.down_blocks):  # 3
            out = block(out)
            encoder_map.append(out)

        deformation = dense_motion[0]
        occlusion_map = dense_motion[1:]

        out = self.deform_input(out, deformation)
        out = self.occlude_input(out, occlusion_map[0])

        for i, up_block in enumerate(self.up_blocks):  # 3
            for j, re_block in enumerate(self.resblock):
                if j == 2 * i or j == 2 * i + 1:
                    out = re_block(out)
            out = up_block(out)
            # i --> [0, 1, 2], -(i + 2) --> [-2, -3, -4]
            encode_i = encoder_map[-(i + 2)]
            encode_i = self.deform_input(encode_i, deformation)
            encode_i = self.occlude_input(encode_i, occlusion_map[i + 1])

            if i < self.num_down_blocks - 1:
                out = torch.cat([out, encode_i], dim=1)

        deformed_source = self.deform_input(source_image, deformation)

        occlusion_last = occlusion_map[-1]
        out = out * (1.0 - occlusion_last) + encode_i
        out = torch.sigmoid(self.final(out))
        out = out * (1.0 - occlusion_last) + deformed_source * occlusion_last

        return out


class ImageAnimation(nn.Module):
    def __init__(self):
        super(ImageAnimation, self).__init__()
        self.kpdetector = KPDetector()
        self.densemotion = DenseMotionNetwork()
        self.generator = InpaintingNetwork()

        # checkpoint_path = "../checkpoints/vox.pth.tar"
        # checkpoint = torch.load(checkpoint_path)
        # self.generator.load_state_dict(checkpoint['inpainting_network'])
        # self.kpdetector.load_state_dict(checkpoint['kp_detector'])
        # self.densemotion.load_state_dict(checkpoint['dense_motion_network'])
        # torch.save(self.state_dict(), "/tmp/image_animation.pth")

    def forward(self, source, driving, first_driving):
        kp_source = self.kpdetector(source)
        kp_driving = self.kpdetector(driving)
        kp_first_driving = self.kpdetector(first_driving)
        kp_source_normal = kp_source + kp_driving - kp_first_driving

        dense_motion = self.densemotion(source, kp_source_normal, kp_source)
        out = self.generator(source, dense_motion)
        return out.clamp(0.0, 1.0)
