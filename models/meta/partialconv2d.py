###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


def partial_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_res=True):
        super(PartialBottleneck, self).__init__()
        self.use_res = use_res
        self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,
                                   dilation=2)
        self.conv3 = PartialConv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(True)
        for name, module in self.group1.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, mask=None):
        x1 = x
        x = self.conv1(x, mask)
        x = self.relu(x)
        x = self.conv2(x, mask)
        x = self.relu(x)
        x = self.conv3(x, mask)
        if self.use_res:
            x = self.relu(x1 + x)
        return x


class PartialBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(PartialBasicBlock, self).__init__()
        self.conv1 = partial_conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = partial_conv3x3(planes, planes)

    def forward(self, x, mask=None):
        residual = x
        out = self.conv1(x, mask)
        out = self.relu(out)
        out = self.conv2(out, mask)
        out += residual
        out = self.relu(out)
        return out
