import numpy as np
import rasterio
import torch

from models.meta.model import TUA_CR
from utils.common import PREPROCESS, POSTPROCESS, ALLPROCESS
from utils.feature_detectors import get_cloud_cloudshadow_mask


def get_preview(file, predicted_file=False, bands=None, brighten_limit=None, sar_composite=False):
    if bands is None:
        bands = [4, 3, 2]
    if not predicted_file:
        with rasterio.open(file) as src:
            r, g, b = src.read(bands)

    else:
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


def get_image(path):
    with rasterio.open(path, 'r', driver='GTiff') as src:
        image = src.read()
        image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
    return image


def build_data(input_path=None, target_path=None, cloudy_path=None, sar_path=None, input_path2=None,
               process_type=ALLPROCESS, input_path3=None):
    result = {}
    target_img = get_image(target_path).astype('float32')
    target_img = get_normalized_data(target_img, 3)
    result['target'] = torch.from_numpy(target_img)
    if process_type == POSTPROCESS or process_type == ALLPROCESS:
        input_img = get_image(input_path).astype('float32')
        input_img2 = get_image(input_path2).astype('float32')
        input_img3 = get_image(input_path3).astype('float32')
        input_img = get_normalized_data(input_img, 2)
        input_img2 = get_normalized_data(input_img2, 4)
        input_img3 = get_normalized_data(input_img3, 4)

        result['input'] = torch.from_numpy(input_img)
        result['input2'] = torch.from_numpy(input_img2)
        result['input3'] = torch.from_numpy(input_img3)

    if process_type == PREPROCESS or process_type == ALLPROCESS:
        cloudy_img_ori = get_image(cloudy_path).astype('float32')
        cloudy_img = get_image(cloudy_path).astype('float32')
        sar_img = get_image(sar_path).astype('float32')
        cloudy_img = get_normalized_data(cloudy_img, 2)
        sar_img = get_normalized_data(sar_img, 1)
        result['cloudy_ori'] = torch.from_numpy(cloudy_img_ori)
        result['cloudy'] = torch.from_numpy(cloudy_img)
        result['sar'] = torch.from_numpy(sar_img)
    return result


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


def get_attention_mask(cloudy_path):
    cloudy_image = get_image(cloudy_path).astype('float32')
    mask = get_cloud_cloudshadow_mask(cloudy_image, 0.2)
    t_mask = mask.copy()
    mask[mask == -1] = 0.0
    t_mask[t_mask != 0] = -1
    t_mask[t_mask == 0] = 1.0
    t_mask[t_mask != 0] = 0.0
    return {'mask': torch.from_numpy(mask.astype('float32')).unsqueeze(0),
            't_mask': torch.from_numpy(t_mask.astype('float32')).unsqueeze(0)}


def load_model(path, ensemble_num=3, device=torch.device('cuda:0')):
    model = TUA_CR(2, 13, 3, ensemble_num=ensemble_num, bilinear=True).to(device)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint, strict=True)
    except:
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except:
            pass
    return model.to(device)
