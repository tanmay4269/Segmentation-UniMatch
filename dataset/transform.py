import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask=None, size=69):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    if mask:
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    if mask:
        mask = mask.crop((x, y, x + size, y + size))
    
    return img, mask


def hflip(img, mask=None, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)    
        if mask:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return img, mask


def normalize(args, cfg, img, mask=None):
    mean_12 = [0.48906528337134253, 0.44708340242505074, 0.41347014026509393]
    std_12 = [0.1776324207170142, 0.1674080609033505, 0.15651418413552973]

    mean_3 = [0.4065893 , 0.28742657, 0.04781261]
    std_3 = [0.17281745, 0.14731901, 0.03354312]

    transforms_list = [
        transforms.ToTensor(),
    ]
    if cfg['use_data_normalization']:
        if args.dataset == 'idx_12':
            mean = mean_12
            std = std_12
        else: 
            mean = mean_3
            std = std_3
        
        transforms_list.append(transforms.Normalize(mean=mean, std=std))

    img = transforms.Compose(transforms_list)(img)

    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask=None, ratio_range=(1,1)):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    if mask:
        mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
