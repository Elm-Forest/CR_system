from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from models.meta.cbam import CBAM
from models.meta.partialconv2d import PartialBasicBlock
from models.meta.rrdb import RRDB
from models.meta.sam import conv3x3, SAM
from models.meta.unet_parts import DoubleConv, Down, OutConv, Up


class Encoder(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(Encoder, self).__init__()
        self.n_channels = in_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, use_cbam=True, use_res=False):
        super(Bottleneck, self).__init__()
        self.use_res = use_res
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if use_norm:
            m['bc1'] = nn.BatchNorm2d(out_channels)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        if use_norm:
            m['bc2'] = nn.BatchNorm2d(out_channels)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.ReLU(True)
        for name, module in self.group1.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.group1(x)
        if self.use_res:
            x = self.relu(x1 + x)
        return x


class Fusion_Blocks(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Fusion_Blocks, self).__init__()
        self.cbam1 = CBAM(in_feat, no_spatial=True)
        self.conv_out = conv3x3(in_feat, out_feat)
        self.relu = nn.ReLU(True)
        init.kaiming_normal_(self.conv_out.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.cbam1(x1) + x1
        x1 = self.relu(x1)
        x1 = self.conv_out(x1)
        x1 = self.relu(x1)
        return x1


class Sar_Translation(nn.Module):
    def __init__(self, in_channels_sar, out_channels, feat=32, num_block=8):
        super(Sar_Translation, self).__init__()
        self.conv_in = conv3x3(in_channels_sar, feat)
        self.res_blocks = nn.Sequential(
            *[Bottleneck(feat, feat, use_cbam=False, use_res=True) for _ in range(num_block)])
        self.conv_out = conv3x3(feat, out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        return x


# class AttnCGAN_CR(nn.Module):
#     def __init__(self, in_channels_sar=2, in_channels_s2=13, out_channels=3, ensemble_num=2, bilinear=True,
#                  feature_c=32, num_rrdb=4):
#         super(AttnCGAN_CR, self).__init__()
#         self.bilinear = bilinear
#         self.sar_trans = Sar_Translation(in_channels_sar=in_channels_sar, out_channels=in_channels_s2)
#         self.encoder_sar = Encoder(in_channels_s2 + in_channels_sar, bilinear)  # official Unet Encoder
#         self.encoder_s2 = Encoder(in_channels_s2 * ensemble_num, bilinear)
#         factor = 2 if bilinear else 1
#         self.up1 = (Up(1024, 512 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64 // factor, bilinear))
#         self.fb1 = Fusion_Blocks(128, 64)
#         self.fb2 = Fusion_Blocks(256, 128)
#         self.fb3 = Fusion_Blocks(512, 256)
#         self.fb4 = Fusion_Blocks(1024, 512)
#         self.fb5 = Fusion_Blocks(2048 // factor, 1024 // factor)
#         self.out_s2 = (OutConv(64 // factor, in_channels_s2))
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_in_reg = nn.Sequential(
#             conv3x3(64 // factor, feature_c),
#             nn.ReLU(True),
#         )
#         init.kaiming_normal_(self.conv_in_reg[0].weight, mode='fan_out', nonlinearity='relu')
#         self.SAM1 = SAM(feature_c, feature_c, 1)
#         self.res_block1 = Bottleneck(feature_c, feature_c)
#         self.res_block2 = Bottleneck(feature_c, feature_c)
#         self.res_block3 = Bottleneck(feature_c, feature_c)
#         self.res_block4 = Bottleneck(feature_c, feature_c)
#         self.res_block5 = Bottleneck(feature_c, feature_c)
#         self.res_block6 = Bottleneck(feature_c, feature_c)
#         self.res_block7 = Bottleneck(feature_c, feature_c)
#         self.res_block8 = Bottleneck(feature_c, feature_c)
#         self.rrdb_blocks = nn.Sequential(*[RRDB(feature_c) for _ in range(num_rrdb)])
#         self.outc = (OutConv(feature_c, out_channels))
#
#     def forward(self, x11, x12, x2, x13):
#         # Translate SAR
#         sar_trans = self.sar_trans(x2)
#         x2 = torch.cat((sar_trans, x2), dim=1)
#         x1 = torch.cat((x11, x12, x13), dim=1)
#
#         # I. feat fusion
#         x11, x12, x13, x14, x15 = self.encoder_s2(x1)
#         x21, x22, x23, x24, x25 = self.encoder_sar(x2)
#         x1 = self.fb1(x11, x21)
#         x2 = self.fb2(x12, x22)
#         x3 = self.fb3(x13, x23)
#         x4 = self.fb4(x14, x24)
#         x5 = self.fb5(x15, x25)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         s2 = self.out_s2(x)
#         x = self.conv_in_reg(x)
#
#         # II. fix ROI
#         Attention1 = self.SAM1(x)
#         out = F.relu(self.res_block1(x) * Attention1 + x)
#         out = F.relu(self.res_block2(out) * Attention1 + out)
#         Attention2 = self.SAM1(out)
#         out = F.relu(self.res_block3(out) * Attention2 + out)
#         out = F.relu(self.res_block4(out) * Attention2 + out)
#         Attention3 = self.SAM1(out)
#         out = F.relu(self.res_block5(out) * Attention3 + out)
#         out = F.relu(self.res_block6(out) * Attention3 + out)
#         out = F.relu(self.res_block7(out) + out)
#         out = F.relu(self.res_block8(out) + out)
#         out = x + out
#
#         # III. fix output
#         out = self.rrdb_blocks(out)
#         out = self.outc(out)
#         return [out, s2, sar_trans, Attention3]


class Sar_Translate(nn.Module):
    def __init__(self, in_channels_sar, out_channels, feat=32, num_block=12):
        super(Sar_Translate, self).__init__()
        self.conv_in = conv3x3(in_channels_sar, feat)
        self.res_blocks = nn.Sequential(
            *[BasicBlock(feat, feat) for _ in range(num_block)])
        self.conv_out = conv3x3(feat, out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        return x


class TUA_CR(nn.Module):
    def __init__(self, in_channels_sar=2, in_channels_s2=13, out_channels=3, ensemble_num=2, bilinear=True,
                 feature_c=32, num_rrdb=4):
        super(TUA_CR, self).__init__()
        self.bilinear = bilinear
        self.sar_trans = Sar_Translate(in_channels_sar=in_channels_sar, out_channels=in_channels_s2)
        self.encoder_sar = Encoder(in_channels_s2 + in_channels_sar, bilinear)  # official Unet Encoder
        self.encoder_s2 = Encoder(in_channels_s2 * ensemble_num, bilinear)
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64 // factor, bilinear))
        self.fb1 = Fusion_Blocks(128, 64)
        self.fb2 = Fusion_Blocks(256, 128)
        self.fb3 = Fusion_Blocks(512, 256)
        self.fb4 = Fusion_Blocks(1024, 512)
        self.fb5 = Fusion_Blocks(2048 // factor, 1024 // factor)
        self.relu = nn.ReLU(inplace=True)
        self.out_s1 = (OutConv(64 // factor, in_channels_s2))
        feature_c = 64 // factor
        self.conv_in_attn = nn.Sequential(
            conv3x3(in_channels_s2, feature_c),
            nn.ReLU(True),
        )
        self.sam = SAM(feature_c, feature_c, 1)
        self.res_block_attn1 = Bottleneck(feature_c, feature_c, use_norm=True)
        self.res_block_attn2 = Bottleneck(feature_c, feature_c, use_norm=True)
        self.res_block_attn3 = Bottleneck(feature_c, feature_c, use_norm=True)
        self.res_block1 = PartialBasicBlock(feature_c, feature_c)
        self.res_block2 = PartialBasicBlock(feature_c, feature_c)
        self.res_block3 = PartialBasicBlock(feature_c, feature_c)
        self.res_block4 = PartialBasicBlock(feature_c, feature_c)
        self.res_block5 = PartialBasicBlock(feature_c, feature_c)
        self.res_block6 = PartialBasicBlock(feature_c, feature_c)
        self.res_block7 = PartialBasicBlock(feature_c, feature_c)
        self.res_block8 = PartialBasicBlock(feature_c, feature_c)
        self.res_block9 = PartialBasicBlock(feature_c, feature_c)
        self.res_block10 = PartialBasicBlock(feature_c, feature_c)
        self.res_block11 = PartialBasicBlock(feature_c, feature_c)
        self.out_s2 = (OutConv(feature_c, out_channels))
        self.rrdb_blocks = nn.Sequential(*[RRDB(feature_c) for _ in range(num_rrdb)])
        self.outc = (OutConv(feature_c, out_channels))

    def forward(self, x11, x12, x2, cloudy):
        # Translate SAR
        sar_trans = self.sar_trans(x2)
        x2 = torch.cat((sar_trans, x2), dim=1)
        x1 = torch.cat((x11, x12, cloudy), dim=1)

        # I. Feat Fusion
        x11, x12, x13, x14, x15 = self.encoder_s2(x1)
        x21, x22, x23, x24, x25 = self.encoder_sar(x2)
        x1 = self.fb1(x11, x21)
        x2 = self.fb2(x12, x22)
        x3 = self.fb3(x13, x23)
        x4 = self.fb4(x14, x24)
        x5 = self.fb5(x15, x25)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        stage1 = self.out_s1(x)

        # II. Fix ROI
        cloudy = self.conv_in_attn(cloudy)
        attn1 = self.sam(cloudy)
        cloudy = F.relu(self.res_block_attn1(cloudy) * attn1 + cloudy)
        cloudy = F.relu(self.res_block_attn2(cloudy) * attn1 + cloudy)
        cloudy = F.relu(self.res_block_attn3(cloudy) * attn1 + cloudy)
        out = self.res_block1(x, attn1)
        out = self.res_block2(out, attn1)
        out = self.res_block3(out, attn1)
        attn2 = self.sam(cloudy)
        cloudy = F.relu(self.res_block_attn1(cloudy) * attn2 + cloudy)
        cloudy = F.relu(self.res_block_attn2(cloudy) * attn2 + cloudy)
        cloudy = F.relu(self.res_block_attn3(cloudy) * attn2 + cloudy)
        out = self.res_block4(out, attn2)
        out = self.res_block5(out, attn2)
        out = self.res_block6(out, attn2)
        attn3 = self.sam(cloudy)
        out = self.res_block7(out, attn3)
        out = self.res_block8(out, attn3)
        out = self.res_block9(out, attn3)
        out = self.res_block10(out)
        out = self.res_block11(out)
        out = x + out
        stage2 = self.out_s2(out)

        # III. Enhance Output
        out = self.rrdb_blocks(out)
        out = self.outc(out)
        return [out, stage2, sar_trans, attn3, stage1]
