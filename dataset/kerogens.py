from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
from PIL import Image
import cv2
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt


class KerogensDataset(Dataset):
    def __init__(
            self, 
            data_root, 
            mode, 
            args,
            cfg, 
            size=None, 
            nsample=None,
        ):
        self.dataset_name = args.dataset
        self.mode = mode
        self.size = size

        self.p_jitter = cfg['p_jitter']
        self.p_gray = cfg['p_gray']
        self.p_blur = cfg['p_blur']
        self.p_cutmix = cfg['p_cutmix']

        self.images_dir = os.path.join(data_root, 'image')
        self.images = sorted(os.listdir(self.images_dir))

        if mode == 'train_u':
            if nsample < len(self.images):
                self.images = self.images[:nsample]
            return
        
        self.masks_dir = os.path.join(data_root, 'label')
        self.masks = sorted(os.listdir(self.masks_dir))

        if mode == 'val' or mode == 'test':
            return

        if nsample:
            self.images *= math.ceil(nsample / len(self.images))
            self.masks *= math.ceil(nsample / len(self.masks))

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])

        img = Image.open(img_path)
        
        mask = None
        if self.mode != 'train_u':
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask_np = np.load(mask_path)
            if self.dataset_name == 'idx_3':
                mask_np = mask_np / 3
            mask = Image.fromarray(mask_np)

        if self.mode == 'val' or self.mode == 'test':
            img, mask = normalize(img, mask)
            return idx, img, mask

        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < self.p_jitter:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=self.p_gray)(img_s1)
        img_s1 = blur(img_s1, p=self.p_blur)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=self.p_cutmix)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        img_s1 = normalize(img_s1)
        img_s2 = normalize(img_s2)

        return normalize(img_w), img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.images)
