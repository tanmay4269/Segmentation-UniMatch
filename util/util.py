import os
import argparse
import wandb 

import numpy as np

import torch.nn.functional as F

enable_logging = False

def init_logging(args, cfg):
    global enable_logging

    enable_logging = args.enable_logging
    print("wandb Logging:", enable_logging)

    if enable_logging:
        wandb.init(project=args.project_name, name=args.model_name, config=cfg)


def log(dict):
    global enable_logging

    if enable_logging:
        wandb.log(dict)


def get_args():
    parser = argparse.ArgumentParser(description='Stranger-Sections-2')
    parser.add_argument('--project_name', type=str, default='ss2-idx_3')
    parser.add_argument('--model_name', type=str, default='debug')
    
    parser.add_argument('--search_alg', type=str, default='rand')
    parser.add_argument('--fast_debug', action='store_true')
    parser.add_argument('--enable_logging', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')  # depriciated 

    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nclass', type=int, required=True)

    parser.add_argument('--num_samples', type=int, required=False)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--epochs_before_eval', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    data_root = args.dataroot
    labeled_data_dir = os.path.join(data_root, f'labeled/train/{args.dataset}')
    val_data_dir = os.path.join(data_root, f'labeled/val/{args.dataset}')
    unlabeled_data_dir = os.path.join(data_root, f'unlabeled/{args.dataset}')
    test_data_dir = os.path.join(data_root, f'test/{args.dataset}')

    parser.add_argument('--labeled_data_dir', type=str, default=labeled_data_dir)
    parser.add_argument('--unlabeled_data_dir', type=str, default=unlabeled_data_dir)
    parser.add_argument('--val_data_dir', type=str, default=val_data_dir)
    parser.add_argument('--test_data_dir', type=str, default=test_data_dir)

    args = parser.parse_args()

    return args


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average, current value, and variance"""

    def __init__(self, length=0, track_variance=False):
        self.length = length
        self.track_variance = track_variance
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
            self.squared_sum = 0.0  # For variance calculation
        self.val = 0.0
        self.avg = 0.0
        self.var = 0.0  # Initialize variance

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1  # Avoid bad usage; refine if needed
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
            if self.track_variance:
                self.var = np.var(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
            if self.track_variance:
                self.squared_sum += (val ** 2) * num
                mean_squared = self.avg ** 2
                mean_of_squares = self.squared_sum / self.count
                self.var = mean_of_squares - mean_squared