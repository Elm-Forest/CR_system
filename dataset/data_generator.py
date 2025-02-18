import os
import os.path

import keras
import numpy as np
import rasterio
import scipy.signal as scisig

from utils.feature_detectors import get_cloud_cloudshadow_mask


class DataGenerator(keras.utils.Sequence):
    """DataGenerator for Keras routines."""

    def __init__(self,
                 list_IDs,
                 batch_size=32,
                 input_dim=((13, 256, 256), (2, 256, 256)),
                 scale=2000,
                 shuffle=True,
                 include_target=True,
                 data_augmentation=False,
                 random_crop=False,
                 crop_size=128,
                 clip_min=None,
                 clip_max=None,
                 input_data_folder='./',
                 use_cloud_mask=True,
                 max_val_sar=5,
                 cloud_threshold=0.2
                 ):

        if clip_min is None:
            clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.nr_images = len(self.list_IDs)
        self.indexes = np.arange(self.nr_images)
        self.scale = scale
        self.shuffle = shuffle
        self.include_target = include_target
        self.data_augmentation = data_augmentation
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.max_val = max_val_sar

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.input_data_folder = input_data_folder
        self.use_cloud_mask = use_cloud_mask
        self.cloud_threshold = cloud_threshold

        self.augment_rotation_param = np.repeat(0, self.nr_images)
        self.augment_flip_param = np.repeat(0, self.nr_images)
        self.random_crop_paramx = np.repeat(0, self.nr_images)
        self.random_crop_paramy = np.repeat(0, self.nr_images)

        self.on_epoch_end()

        print("Generator initialized")

    def __len__(self):
        """Gets the number of batches per epoch"""
        return int(np.floor(self.nr_images / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch from shuffled indices list
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.include_target:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                          self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                          self.random_crop_paramy[indexes])
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                       self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                       self.random_crop_paramy[indexes])
            return X

    def on_epoch_end(self):
        """Update indexes after each epoch."""

        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.data_augmentation:
            self.augment_rotation_param = np.random.randint(0, 4, self.nr_images)
            self.augment_flip_param = np.random.randint(0, 3, self.nr_images)

        if self.random_crop:
            self.random_crop_paramx = np.random.randint(0, self.crop_size, self.nr_images)
            self.random_crop_paramy = np.random.randint(0, self.crop_size, self.nr_images)
        return

    def __data_generation(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                          random_crop_paramx_temp, random_crop_paramy_temp):

        input_opt_batch, cloud_mask_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp,
                                                           augment_flip_param_temp, random_crop_paramx_temp,
                                                           random_crop_paramy_temp, data_type=3)

        input_sar_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                         random_crop_paramx_temp, random_crop_paramy_temp, data_type=1)

        if self.include_target:
            output_opt_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                              random_crop_paramx_temp, random_crop_paramy_temp, data_type=2)
            if self.use_cloud_mask > 0:
                output_opt_cloud_batch = [np.append(output_opt_batch[sample], cloud_mask_batch[sample], axis=0) for
                                          sample in range(len(output_opt_batch))]
                output_opt_cloud_batch = np.asarray(output_opt_cloud_batch)
                return ([input_opt_batch, input_sar_batch], [output_opt_cloud_batch])
            else:
                return ([input_opt_batch, input_sar_batch], [output_opt_batch])
        elif not self.include_target:
            # for prediction step where target is predicted
            return ([input_opt_batch, input_sar_batch])

    def get_image_data(self, paramx, paramy, path):
        # with block not working with window kw
        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read(window=((paramx, paramx + self.crop_size), (paramy, paramy + self.crop_size)))
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image

    def get_opt_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        return image.astype('float32')

    def get_sar_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        medianfilter_onsar = False
        if medianfilter_onsar:
            image[0] = scisig.medfilt2d(image[0], 7)
            image[1] = scisig.medfilt2d(image[1], 7)

        return image.astype('float32')

    def get_data_image(self, ID, data_type, paramx, paramy):

        data_path = os.path.join(self.input_data_folder, ID[data_type], ID[4]).lstrip()

        if data_type == 2 or data_type == 3:
            data_image = self.get_opt_image(data_path, paramx, paramy)
        elif data_type == 1:
            data_image = self.get_sar_image(data_path, paramx, paramy)
        else:
            print('Error! Data type invalid')

        return data_image

    def get_normalized_data(self, data_image, data_type):

        shift_data = False

        shift_values = [[0, 0], [1300., 981., 810., 380., 990., 2270., 2070., 2140., 2200., 650., 15., 1600., 680.],
                        [1545., 1212., 1012., 713., 1212., 2476., 2842., 2775., 3174., 546., 24., 1776., 813.]]

        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
            if shift_data:
                data_image -= self.max_val / 2
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                if shift_data:
                    data_image[channel] -= shift_values[data_type - 1][channel]

            data_image /= self.scale

        return data_image

    def get_batch(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp, random_crop_paramx_temp,
                  random_crop_paramy_temp, data_type):

        if data_type == 1:
            dim = self.input_dim[1]
        else:
            dim = self.input_dim[0]

        batch = np.empty((self.batch_size, *dim)).astype('float32')
        cloud_mask_batch = np.empty((self.batch_size, self.input_dim[0][1], self.input_dim[0][2])).astype('float32')

        for i, ID in enumerate(list_IDs_temp):

            data_image = self.get_data_image(ID, data_type, random_crop_paramx_temp[i], random_crop_paramy_temp[i])
            if self.data_augmentation:
                if not augment_flip_param_temp[i] == 0:
                    data_image = np.flip(data_image, augment_flip_param_temp[i])
                if not augment_rotation_param_temp[i] == 0:
                    data_image = np.rot90(data_image, augment_rotation_param_temp[i], (1, 2))

            if data_type == 3 and self.use_cloud_mask:
                cloud_mask = get_cloud_cloudshadow_mask(data_image, self.cloud_threshold)
                cloud_mask[cloud_mask != 0] = 1
                cloud_mask_batch[i,] = cloud_mask

            data_image = self.get_normalized_data(data_image, data_type)

            batch[i,] = data_image

        cloud_mask_batch = cloud_mask_batch[:, np.newaxis, :, :]

        if data_type == 3:
            return batch, cloud_mask_batch
        else:
            return batch
