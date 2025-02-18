from math import exp

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=5, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def getRGBImg(r, g, b, img_size=256):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)

        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out


def Get2Img(img_fake, img_truth, img_size=256):
    output_img = np.zeros((img_size, 2 * img_size, 3), dtype=np.uint8)
    img_fake = uint16to8((torch.squeeze(img_fake).cpu().detach().numpy() * 10000).astype("uint16")).transpose(1, 2, 0)
    img_truth = uint16to8((torch.squeeze(img_truth).cpu().detach().numpy() * 10000).astype("uint16")).transpose(1, 2, 0)

    img_fake_RGB = getRGBImg(img_fake[:, :, 2], img_fake[:, :, 1], img_fake[:, :, 0], img_size)
    img_truth_RGB = getRGBImg(img_truth[:, :, 2], img_truth[:, :, 1], img_truth[:, :, 0], img_size)

    output_img[:, 0 * img_size:1 * img_size, :] = img_fake_RGB
    output_img[:, 1 * img_size:2 * img_size, :] = img_truth_RGB
    return output_img


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_tensor_to_geotiff(tensor, path_out, dtype=rasterio.float32):
    """
    将 PyTorch 张量保存为 GeoTIFF 文件。

    参数:
    - tensor: 要保存的图像张量，假设形状为 (Channels, Height, Width)。
    - path_out: 输出 GeoTIFF 文件的路径。
    - transform: 仿射变换参数，用于定义像素与地理空间坐标之间的关系。
    - crs: 坐标参考系统。
    - dtype: 数据类型，默认为 rasterio.float32。
    """
    # tensor_np = tensor.cpu().detach().numpy().astype(np.float32)
    tensor_np = tensor.astype(np.float32)
    # 确保张量是期望的形状 (Channels, Height, Width)
    if tensor_np.ndim != 3:
        raise ValueError("张量形状应为 (Channels, Height, Width)")

    # 定义图像的元数据
    meta = {
        'driver': 'GTiff',  # GeoTIFF 格式
        'dtype': dtype,
        'nodata': None,  # 如果有需要处理的无数据值，可以在这里指定
        'width': tensor_np.shape[2],
        'height': tensor_np.shape[1],
        'count': tensor_np.shape[0],  # 波段数量
    }

    # 写入 GeoTIFF 文件
    with rasterio.open(path_out, 'w', **meta) as dst:
        dst.write(tensor_np)
