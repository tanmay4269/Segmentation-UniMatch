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
import albumentations as A


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

        self.args = args
        self.cfg = cfg

        self.images_dir = os.path.join(data_root, 'image')
        self.images = sorted(os.listdir(self.images_dir))

        if mode == 'train_u':
            self.p_jitter_u = cfg['p_jitter_u']
            self.p_gray_u = cfg['p_gray_u']
            self.p_blur_u = cfg['p_blur_u']
            self.p_cutmix_u = cfg['p_cutmix_u']

            if nsample and nsample < len(self.images):
                self.images = self.images[:nsample]
            return
        
        if mode == 'test':
            return

        self.masks_dir = os.path.join(data_root, 'label')
        self.masks = sorted(os.listdir(self.masks_dir))

        if mode == 'val':
            return

        self.p_jitter_l = cfg['p_jitter_l']
        self.p_gray_l = cfg['p_gray_l']
        self.p_blur_l = cfg['p_blur_l']
        self.p_cutmix_l = cfg['p_cutmix_l']

        if nsample and nsample > len(self.images):
            self.images *= math.ceil(nsample / len(self.images))
            self.masks *= math.ceil(nsample / len(self.masks))

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])

        img = Image.open(img_path)
        
        mask = None
        if self.mode != 'train_u' and self.mode != 'test':
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask_np = np.load(mask_path)
            if self.dataset_name == 'idx_3':
                mask_np = mask_np / 3
            mask = Image.fromarray(mask_np)

        if self.mode == 'test':
            img = normalize(self.args, self.cfg, img)
            _, img_file = os.path.split(img_path)
            file_name, _ = os.path.splitext(img_file)
            return file_name, img
        
        if self.mode == 'val':
            _img, _mask = normalize(self.args, self.cfg, img, mask)
            return idx, _img, _mask
        
        img, mask = random_flip(img, mask)
        img, mask = random_rotate(img, mask)
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)

        if self.mode == 'train_l':
            aug_img = deepcopy(img)
            
            cutmix_box = obtain_cutmix_box(aug_img.size[0], p=self.p_cutmix_l)
            
            aug_img = np.array(aug_img)

            if random.random() < self.p_jitter_l:
                l = 0.1; m = 0.25
                aug_img = A.Compose([
                    A.ColorJitter(brightness=(l, m), contrast=(l, m), saturation=(l, m), hue=(l/2, m/2)),
                ])(image=aug_img)['image']

            aug_img = A.Compose([A.ToGray(p=self.p_gray_l)])(image=aug_img)['image']
            aug_img = A.Compose([A.Blur(blur_limit=3, p=self.p_blur_l)])(image=aug_img)['image']


            _img, _mask = normalize(self.args, self.cfg, aug_img, mask)
            return _img, _mask, cutmix_box

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < self.p_jitter_u:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=self.p_gray_u)(img_s1)
        img_s1 = blur(img_s1, p=self.p_blur_u)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=self.p_cutmix_u)

        if random.random() < self.p_jitter_u:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=self.p_gray_u)(img_s2)
        img_s2 = blur(img_s2, p=self.p_blur_u)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=self.p_cutmix_u)

        img_s1 = normalize(self.args, self.cfg, img_s1)
        img_s2 = normalize(self.args, self.cfg, img_s2)

        return  normalize(self.args, self.cfg, img_w), img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.images)
