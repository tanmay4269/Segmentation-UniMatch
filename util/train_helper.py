import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def init_model(nclass, backbone, checkpoint_path=None):
    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=nclass,
    )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])

    return model.cuda()


def init_optimizer(model, cfg):
    optimizer = Adam(
            [
                {
                    'params': model.encoder.parameters(), 
                    'lr': cfg['lr']
                },
                {
                    'params': [param for name, param in model.named_parameters() if 'encoder' not in name], 
                    'lr': cfg['lr'] * cfg['lr_multi']
                }
            ], 
            lr=cfg['lr'], 
            weight_decay=cfg['weight_decay']
        )
    
    return optimizer

def get_mean_std_classweights(loader, args):
    mean = np.zeros(3)
    std = np.zeros(3)

    if args.nclass == 1:
        total_counts = np.zeros(args.nclass + 1)
    else:
        total_counts = np.zeros(args.nclass)

    for img, mask in loader:
        mean += torch.mean(img, dim=(0, 2, 3)).numpy()
        std += torch.std(img, dim=(0, 2, 3)).numpy()

        unique, counts = torch.unique(mask, return_counts=True)

        for i, c in zip(unique, counts):
            total_counts[i] += c.item()


    mean /= len(loader)
    std /= len(loader)

    grand_count = sum(total_counts)
    class_weights = [grand_count / count for count in total_counts]
    normalized_cw = class_weights / max(class_weights)

    return mean.tolist(), std.tolist(), normalized_cw.tolist()