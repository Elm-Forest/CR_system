import argparse
import json
import os
import pprint
import sys

import numpy as np
import rasterio
import torch

from models.UnCRtainTS.parse_args import create_parser

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from models.UnCRtainTS.src import utils
from models.UnCRtainTS.src.model_utils import get_model, load_checkpoint
from models.UnCRtainTS.train_reconstruct import prepare_output, seed_packages

parser = create_parser(mode='test')
test_config = parser.parse_args()

# grab the PID so we can look it up in the logged config for server-side process management
test_config.pid = os.getpid()

# related to flag --use_custom:
# define custom target S2 patches (these will be mosaiced into a single sample), and fetch associated target S1 patches as well as input data
# (TODO: keeping this hard-coded until a more convenient way to pass it as an argument comes about ...)
targ_s2 = [f'ROIs1868/73/S2/14/s2_ROIs1868_73_ImgNo_14_2018-06-21_patch_{pdx}.tif' for pdx in
           [171, 172, 173, 187, 188, 189, 203, 204, 205]]

# load previous config from training directories

# if no custom path to config file is passed, try fetching config file at default location
conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name,
                         "conf.json") if not test_config.load_config else test_config.load_config
if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = ['pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder', 'root1', 'root2',
                        'root3',
                        'max_samples_count', 'batch_size', 'display_step', 'plot_every', 'export_every', 'input_t',
                        'region', 'min_cov', 'max_cov']
        conf_dict = {key: val for key, val in model_config.items() if key not in no_overwrite}
        for key, val in vars(test_config).items():
            if key in no_overwrite: conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
else:
    config = test_config  # otherwise, keep passed flags without any overwriting
config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
device = torch.device(config.device)
if config.pretrain: config.batch_size = 32

experime_dir = os.path.join(config.res_dir, config.experiment_name)
if not os.path.exists(experime_dir): os.makedirs(experime_dir)
with open(os.path.join(experime_dir, "conf.json"), "w") as file:
    file.write(json.dumps(vars(config), indent=4))

# seed everything
seed_packages(config.rdm_seed)
if __name__ == "__main__": pprint.pprint(config)


def get_preview(file, predicted_file=False, bands=None, brighten_limit=None, sar_composite=False):
    if bands is None:
        bands = [4, 3, 2]
    if not predicted_file:
        # with rasterio.open(file, 'r', driver='GTiff') as src:
        #     r, g, b = src.read(bands)
        with rasterio.open(file, 'r', driver='GTiff') as src:
            image = src.read()
        r = image[3]
        g = image[2]
        b = image[1]
    else:
        # original_channels_output = file[:13, :, :]  # 提取前13个通道
        # from Code.tools.tools import save_tensor_to_geotiff
        # save_tensor_to_geotiff(original_channels_output, path_out='K://test//gg.tif')
        # file is actually the predicted array
        r = file[bands[0] - 1]
        g = file[bands[1] - 1]
        b = file[bands[2] - 1]

    if brighten_limit is None:
        return get_rgb_preview(r, g, b, brighten_limit=None, sar_composite=sar_composite)
    else:
        r = np.clip(r, 0, brighten_limit)
        g = np.clip(g, 0, brighten_limit)
        b = np.clip(b, 0, brighten_limit)
        return get_rgb_preview(r, g, b, brighten_limit=None, sar_composite=sar_composite)


def get_image(path):
    with rasterio.open(path, 'r', driver='GTiff') as src:
        image = src.read()
        image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
    return image


def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img = (img - oldMin) / oldRange
    return img


def process_MS(img, method='default'):
    if method == 'default':
        intensity_min, intensity_max = 0, 10000  # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)  # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min,
                      intensity_max)  # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method == 'resnet':
        intensity_min, intensity_max = 0, 10000  # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)  # intensity clipping to a global unified MS intensity range
        img /= 2000  # project to [0,5], preserve global intensities (across patches)
    img = np.nan_to_num(img)
    return img


def get_normalized_data(data_image, data_type=2):
    clip_min = [[-25.0, -32.5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    clip_max = [[0, 0],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]
    max_val = 1
    scale = 10000
    # SAR
    if data_type == 1:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                          clip_max[data_type - 1][channel])
            data_image[channel] -= clip_min[data_type - 1][channel]
            data_image[channel] = max_val * (data_image[channel] / (
                    clip_max[data_type - 1][channel] - clip_min[data_type - 1][channel]))
    # OPT
    elif data_type == 2 or data_type == 3:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                          clip_max[data_type - 1][channel])
        data_image /= scale
    elif data_type == 4:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                          clip_max[data_type - 1][channel])
    return data_image


def process_SAR(img, method='default'):
    if method == 'default':
        dB_min, dB_max = -25, 0  # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)  # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min,
                      dB_max)  # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method == 'resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate(
            [(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
             (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    img = np.nan_to_num(img)
    return img


def build_data(input_path, target_path, cloudy_path, sar_path, input_path2):
    target_img = get_image(target_path).astype('float32')
    cloudy_img_ori = get_image(cloudy_path).astype('float32')
    cloudy_img = get_image(cloudy_path).astype('float32')
    sar_img = get_image(sar_path).astype('float32')
    target_img = get_normalized_data(target_img, 3)
    cloudy_img = process_MS(cloudy_img)
    sar_img = process_SAR(sar_img)
    return {'target': torch.from_numpy(target_img),
            'cloudy_ori': torch.from_numpy(cloudy_img_ori),
            'cloudy': torch.from_numpy(cloudy_img),
            'sar': torch.from_numpy(sar_img)}


def get_rgb_preview(r, g, b, brighten_limit=None, sar_composite=False):
    if brighten_limit is not None:
        r = np.clip(r, 0, brighten_limit)
        g = np.clip(g, 0, brighten_limit)
        b = np.clip(b, 0, brighten_limit)

    if not sar_composite:

        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0:
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)

        return rgb.astype(np.uint8)

    else:
        # generate SAR composite
        HH = r
        HV = g

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6

        rgb = np.dstack((np.zeros_like(HH), HH, HV))

        return rgb.astype(np.uint8)


def pred(config, cloudy, sar, targets, model=None):
    prepare_output(config)
    if model is None:
        model = get_model(config)
        model = model.to(device)
        config.N_params = utils.get_ntrainparams(model)
        ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
        load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")
    model.eval()
    x = cloudy.to(device).unsqueeze(0)
    if config.use_sar:
        x = torch.cat((sar.to(device).unsqueeze(0), x), dim=1).unsqueeze(0)
    y = targets.to(device).unsqueeze(0).unsqueeze(0)
    inputs = {'A': x, 'B': y, 'dates': None, 'masks': None}
    model.set_input(inputs)
    model.forward()
    model.get_loss_G()
    model.rescale()
    outputs = model.fake_B.squeeze(0)
    return outputs


def get_uncrtain():
    model = get_model(config)
    model = model.to(device)
    config.N_params = utils.get_ntrainparams(model)
    ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
    load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")
    model.eval()
    return model


def pred_uncrtain(cloudy, sar, targets, model=None):
    return pred(config, cloudy, sar, targets, model)
