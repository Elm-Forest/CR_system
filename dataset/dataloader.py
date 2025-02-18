import os
import random

import numpy as np
import rasterio
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset

import csv
from utils.feature_detectors import get_cloud_cloudshadow_mask


def get_filelists(listpath):
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)
    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)
    csv_file.close()
    return train_filelist, val_filelist, test_filelist


class SEN12MSCR_Dataset(Dataset):
    def __init__(self, filelist, inputs_dir, targets_dir, sar_dir=None, inputs_dir2=None, crop_size=None,
                 use_attention=False, cloudy_dir=None):
        self.filelist = filelist
        self.inputs_dir = inputs_dir
        self.inputs_dir2 = inputs_dir2
        self.sar_dir = sar_dir
        self.targets_dir = targets_dir
        self.cloudy_dir = cloudy_dir
        self.crop_size = crop_size
        self.use_attention = use_attention
        self.clip_min = [[-25.0, -32.5],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fileID = self.filelist[index][-1]

        input_path = os.path.join(self.inputs_dir, fileID)
        target_path = os.path.join(self.targets_dir, fileID)

        input_image = self.get_image(input_path).astype('float32')
        target_image = self.get_image(target_path).astype('float32')

        input_image_np = self.get_normalized_data(input_image, data_type=2)
        input_image = torch.from_numpy(input_image_np)
        target_image_np = self.get_normalized_data(target_image, data_type=3)
        target_image = torch.from_numpy(target_image_np)
        x, y = 0, 0
        if self.crop_size is not None:
            y = random.randint(0, np.maximum(0, self.crop_size))
            x = random.randint(0, np.maximum(0, self.crop_size))
            target_image = target_image[..., y:y + self.crop_size, x:x + self.crop_size]
            input_image = input_image[..., y:y + self.crop_size, x:x + self.crop_size]
        result = {'input': input_image,
                  'target': target_image}

        if self.sar_dir is not None:
            sar_path = os.path.join(self.sar_dir, fileID)
            sar_image = self.get_image(sar_path).astype('float32')
            sar_image = self.get_normalized_data(sar_image, data_type=1)
            sar_image = torch.from_numpy(sar_image)
            if self.crop_size is not None:
                sar_image = sar_image[..., y:y + self.crop_size, x:x + self.crop_size]
            result['sar'] = sar_image
        if self.inputs_dir2 is not None:
            input_path2 = os.path.join(self.inputs_dir2, fileID)
            input_image2 = self.get_image(input_path2).astype('float32')
            input_image2_np = self.get_normalized_data(input_image2, data_type=4)
            input_image2 = torch.from_numpy(input_image2_np)
            if self.crop_size is not None:
                input_image2 = input_image2[..., y:y + self.crop_size, x:x + self.crop_size]
            result['input2'] = input_image2
            if self.cloudy_dir is not None:
                cloudy_path = os.path.join(self.cloudy_dir, fileID)
                cloudy_image = self.get_image(cloudy_path).astype('float32')
                cloudy_image_n = self.get_normalized_data(cloudy_image, data_type=3)
                if self.crop_size is not None:
                    cloudy_image = cloudy_image[..., y:y + self.crop_size, x:x + self.crop_size]
                    cloudy_image_n = cloudy_image_n[..., y:y + self.crop_size, x:x + self.crop_size]
                result['cloudy'] = torch.from_numpy(cloudy_image_n)

                if self.use_attention:
                    result['mask'], result["t_mask"] = self.get_attention_mask(cloudy=cloudy_image)
                    # result['attention'] = torch.from_numpy(
                    #     self.get_attention_map(input1=input_image_np, targets=target_image_np, input2=input_image2_np))
        else:
            if self.cloudy_dir is not None:
                cloudy_path = os.path.join(self.cloudy_dir, fileID)
                cloudy_image = self.get_image(cloudy_path).astype('float32')
                cloudy_image_n = self.get_normalized_data(cloudy_image, data_type=3)
                if self.crop_size is not None:
                    cloudy_image = cloudy_image[..., y:y + self.crop_size, x:x + self.crop_size]
                    cloudy_image_n = cloudy_image_n[..., y:y + self.crop_size, x:x + self.crop_size]
                result['cloudy'] = torch.from_numpy(cloudy_image_n)
                if self.use_attention:
                    result['mask'], result["t_mask"] = self.get_attention_mask(cloudy=cloudy_image)
                # result['attention'] = torch.from_numpy(
                #     self.get_attention_map(input1=input_image_np, targets=target_image_np))
        return result

    def get_image(self, path):
        with rasterio.open(path, 'r', driver='GTiff') as src:
            image = src.read()
            image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
        return image

    # def get_normalized_data(self, data_image):
    #     data_image = np.clip(data_image, self.clip_min, self.clip_max)
    #     data_image = data_image / self.scale
    #     return data_image
    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
            data_image /= self.scale
        elif data_type == 4:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
        return data_image

    def get_attention_mask(self, cloudy=None):
        mask = get_cloud_cloudshadow_mask(cloudy, 0.2)
        t_mask = mask.copy()
        mask[mask == -1] = 0.5
        t_mask[t_mask != 0] = -1
        t_mask[t_mask == 0] = 1.0
        t_mask[t_mask != 0] = 0.0
        return (torch.from_numpy(mask.astype('float32')).unsqueeze(0),
                torch.from_numpy(t_mask.astype('float32')).unsqueeze(0))

    def get_attention_map(self, input1, targets, input2=None):
        x = input1
        t = targets
        inputs_R = x[3]
        inputs_G = x[2]
        inputs_B = x[1]
        targets_R = t[3]
        targets_G = t[2]
        targets_B = t[1]
        x = self.get_rgb_preview(inputs_R, inputs_G, inputs_B,
                                 brighten_limit=2000)
        t = self.get_rgb_preview(targets_R, targets_G, targets_B,
                                 brighten_limit=2000)
        x_gray = np.mean(x, axis=2).astype(np.float32)
        t_gray = np.mean(t, axis=2).astype(np.float32)
        _, M_ssim_x = ssim(t_gray, x_gray, full=True, data_range=255.0, win_size=7)
        if input2 is not None:
            y = input2
            inputs2_R = y[3]
            inputs2_G = y[2]
            inputs2_B = y[1]
            y = self.get_rgb_preview(inputs2_R, inputs2_G, inputs2_B,
                                     brighten_limit=2000)
            y_gray = np.mean(y, axis=2).astype(np.float32)
            _, M_ssim_y = ssim(t_gray, y_gray, full=True, data_range=255.0, win_size=7)
            M_ssim = (M_ssim_x + M_ssim_y) * 0.5
            M_ssim = (M_ssim - M_ssim.min()) / (M_ssim.max() - M_ssim.min())
            attn_map = 1 - M_ssim
        else:
            M_ssim = M_ssim_x
            M_ssim = (M_ssim - M_ssim.min()) / (M_ssim.max() - M_ssim.min())
            attn_map = 1 - M_ssim
        return attn_map

    def get_rgb_preview(self, r, g, b, brighten_limit=None, sar_composite=False):
        if brighten_limit is not None:
            r = np.clip(r, 0, brighten_limit)
            g = np.clip(g, 0, brighten_limit)
            b = np.clip(b, 0, brighten_limit)

        if not sar_composite:
            rgb = np.dstack((r, g, b))
            rgb = rgb - np.nanmin(rgb)
            if np.nanmax(rgb) == 0:
                rgb = 255 * np.ones_like(rgb)
            else:
                rgb = 255 * (rgb / np.nanmax(rgb))
            rgb[np.isnan(rgb)] = np.nanmean(rgb)
            return rgb.astype(np.uint8)

        else:
            HH = r
            HV = g
            HH = np.clip(HH, -25.0, 0)
            HH = (HH + 25.1) * 255 / 25.1
            HV = np.clip(HV, -32.5, 0)
            HV = (HV + 32.6) * 255 / 32.6
            rgb = np.dstack((np.zeros_like(HH), HH, HV))
            return rgb.astype(np.uint8)
