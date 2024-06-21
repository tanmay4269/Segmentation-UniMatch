import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from scipy import ndimage


def random_flip(img, mask, p=0.5):
    if random.random() > p:
        return img, mask

    dir = Image.FLIP_LEFT_RIGHT if random.random() < 0.5 else Image.FLIP_TOP_BOTTOM

    img = img.transpose(dir)
    
    if mask: 
        mask = mask.transpose(dir)
    
    return img, mask


def random_rotate(img, mask, p=0.5):
    if random.random() > p:
        return img, mask

    angle = random.uniform(-180, 180)
    img = img.rotate(angle, resample=Image.NEAREST)
    
    if mask:
        mask = mask.rotate(angle, resample=Image.NEAREST)
    
    return img, mask


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


def resize(img, mask, ratio_range=(1,1)):
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


def normalize(args, cfg, img, mask=None):
    mean = {
        'idx_12': {
            'labeled': [0.48818670792712104, 0.4457195938461357, 0.41172934629850916],
            'unlabeled': [0.4242805863420169, 0.38311692674954734, 0.3415233268505997],
            'validation': [0.38173748221662307, 0.3524048891332414, 0.33767416410975987],
        },
        'idx_3': {
            'labeled': [0.40658929619719003, 0.2874265715041581, 0.04781260547392509],
            'unlabeled': [0.5459248080849648, 0.4083577454966657, 0.06468971892095664],
            'validation': [0.36764958687126637, 0.2415672168135643, 0.03286071598995477],
        },
    }

    std = {
        'idx_12': {
            'labeled': [0.1824450346744723, 0.17185980284379587, 0.1603963966998789],
            'unlabeled': [0.16966962694293922, 0.16369790828062428, 0.1570306455095609],
            'validation': [0.160264586408933, 0.15908396492401758, 0.1536410434378518],
        },
        'idx_3': {
            'labeled': [0.17281745322706069, 0.14731901198187294, 0.03354311751420883],
            'unlabeled': [0.24796055835836073, 0.22987997237154667, 0.04662432360517628],
            'validation': [0.08467969996854663, 0.07784020388498902, 0.017431323300115764],
        },
    }

    transforms_list = [
        transforms.ToTensor(),
    ]

    if cfg['data_normalization'] != 'none':
        mean = mean[args.dataset][cfg['data_normalization']]
        std = std[args.dataset][cfg['data_normalization']]
        
        transforms_list.append(transforms.Normalize(mean=mean, std=std))

    img = transforms.Compose(transforms_list)(img)

    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img
