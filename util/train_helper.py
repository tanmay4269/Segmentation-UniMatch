import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Adam

import segmentation_models_pytorch as smp


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def init_model(args, cfg, checkpoint_path=None):
    weights = 'imagenet' if cfg['pretrained'] else None

    model = smp.DeepLabV3Plus(
        encoder_name=cfg['backbone'],
        encoder_weights=weights,
        in_channels=3,
        classes=args.nclass,
    )

    if not cfg['pretrained']:
        model.apply(init_weights)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])

    return model.cuda()


def init_optimizer(model, cfg):
    if not cfg['pretrained']:
        return Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

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


class LossFn(nn.Module):
    def __init__(self, args, cfg):
        super(LossFn, self).__init__()
        
        self.loss_fn = cfg['loss_fn']
        
        if args.nclass > 1:
            cw_array = np.array(cfg['class_weights'])
            cw_array[2] = cfg['class_weights_idx_2']
            class_weights = torch.tensor(cw_array).float().cuda()

        if args.nclass == 1:
            self.criterion_ce = nn.BCEWithLogitsLoss()
            self.criterion_j = smp.losses.JaccardLoss('binary')
        else:
            self.criterion_ce = nn.CrossEntropyLoss(class_weights)
            self.criterion_j = smp.losses.JaccardLoss('multiclass')

    def forward(self, pred, target):
        if self.loss_fn == 'cross_entropy':
            return self.criterion_ce(pred, target)
        elif self.loss_fn == 'jaccard':
            return self.criterion_j(pred, target)
        elif self.loss_fn == 'combined':
            return (self.criterion_ce(pred, target) + self.criterion_j(pred, target)) / 2.0
        

def get_mean_std_classweights(loader, args):
    mean = np.zeros(3)
    std = np.zeros(3)

    if args.nclass == 1:
        total_counts = np.zeros(args.nclass + 1)
    else:
        total_counts = np.zeros(args.nclass)

    for data in loader:
        img = data[0]
        if loader.dataset.mode == 'val':
            img = data[1]

        mean += torch.mean(img, dim=(0, 2, 3)).numpy()
        std += torch.std(img, dim=(0, 2, 3)).numpy()


        # mask = data[1]
        # if loader.dataset.mode == 'val':
        #     mask = data[2]

        # unique, counts = torch.unique(mask, return_counts=True)

        # for i, c in zip(unique, counts):
        # total_counts[i] += c.item()


    mean /= len(loader)
    std /= len(loader)

    # grand_count = sum(total_counts)
    # class_weights = [grand_count / count for count in total_counts]
    # normalized_cw = class_weights / max(class_weights)

    return mean.tolist(), std.tolist() # , normalized_cw.tolist()