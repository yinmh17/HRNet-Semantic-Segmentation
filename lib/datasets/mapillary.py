# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class Mapillary(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=65,
                 multi_scale=True,
                 flip=True,
                 ignore_label=65,
                 base_size=2048,
                 crop_size=(1024, 2048),
                 downsample_rate=1,
                 scale_factor=16,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(Mapillary, self).__init__(ignore_label, base_size,
                                        crop_size, downsample_rate,
                                        scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        # annotation from mapillary, RGB
        self.colors = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128], [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]
        # 'hash' to 1
        colors = np.array(self.colors, dtype=np.float32)
        self.hash_colors = colors[:, 0] + colors[:, 1] * 100 + colors[:, 2] * 10000
        self.label_mapping = {k: i for i, k in enumerate(self.hash_colors)}

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item[:2]
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {"img": image_path,
                      "label": label_path,
                      "name": name, }
            files.append(sample)
        return files

    def convert_label(self, label, inverse=False):
        temp = np.array(label, dtype=np.float32)
        temp = temp[:, :, 0] * 10000 + temp[:, :, 1] * 100 + temp[:, :, 2]
        label = np.zeros_like(temp, dtype=np.uint8)
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'mapillary', item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        label = cv2.imread(os.path.join(self.root, 'mapillary', item["label"]),
                           cv2.IMREAD_COLOR)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip,
                                       self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name
