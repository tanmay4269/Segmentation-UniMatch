# General
import os
import tempfile
from pathlib import Path

import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# smp
import segmentation_models_pytorch as smp

# Data
from torch.utils.data import DataLoader
from dataset.kerogens import KerogensDataset

# Utils
from util.util import *
from util.train_helper import *
from util.eval_helper import *

# Ray
from ray.train import Checkpoint

globally_best_iou = 0

# Set the random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(args, cfg, nsample=None):
    # Datasets    
    trainset_l = KerogensDataset(
        args.labeled_data_dir, 'train_l', 
        args, cfg,
        cfg['crop_size'], nsample
    )

    valset = KerogensDataset(
        args.val_data_dir, 'val',
        args, cfg
    )

    # Dataloaders
    trainloader_l1 = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], shuffle=False,
        pin_memory=True, num_workers=1, drop_last=True
    )

    trainloader_l2 = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], shuffle=True,
        pin_memory=True, num_workers=1, drop_last=True
    )

    valloader = DataLoader(
        valset, batch_size=1, 
        pin_memory=True, num_workers=1, drop_last=False
    )
    
    return trainloader_l1, trainloader_l2, valloader


def trainer(ray_train, args, cfg):
    global globally_best_iou
    set_seed(42)

    print(cfg)

    model_name = str(args.model_name)

    # thresh = cfg['output_thresh']
    # model_name += f'-t{thresh}'

    # if cfg['pretrained']:
    #     model_name += '-pretrained'
    # else:
    #     model_name += '-no-pretraining'

    # j = cfg['p_jitter_l']
    # g = cfg['p_gray_l']
    # b = cfg['p_blur_l']
    # c = cfg['p_cutmix_l']
    # model_name += f'-j{j}-g{g}-b{b}-c{c}'

    # b = cfg['p_blur_l']
    # lr = cfg['lr']

    # model_name += f'-b{b}-l{lr:.1e}'

    args.model_name = model_name
    cfg['save_path'] = args.save_path

    init_logging(args, cfg)

    model = init_model(args, cfg)
    print(f"Param count: {count_params(model):.1f}M")
    optimizer = init_optimizer(model, cfg)

    loss_fn = LossFn(args, cfg)

    num_labeled = len(os.listdir(os.path.join(args.labeled_data_dir, 'label')))
    num_labeled_batches = num_labeled // cfg['batch_size']

    trainloader_l1, trainloader_l2, valloader = load_data(args, cfg)

    total_iters = len(trainloader_l1) * args.num_epochs

    locally_best_iou = 0
    epoch = -1

    total_loss  = AverageMeter(track_variance=True)

    print("Starting Training...")
    while epoch < args.num_epochs:
        loader = zip(trainloader_l1, trainloader_l2)

        for i, ((img_x1, mask_x1, cutmix_box),
                (img_x2, mask_x2, _)) in enumerate(loader):

            if epoch + 1 >= args.num_epochs:
                epoch += 1
                break

            if i % num_labeled_batches == 0:
                epoch += 1

                print(f"Epoch [{epoch}/{args.num_epochs}]\t Previous Best IoU: {locally_best_iou}")

            img_x1, mask_x1, cutmix_box = img_x1.cuda(), mask_x1.cuda(), cutmix_box.cuda()
            img_x2, mask_x2 = img_x2.cuda(), mask_x2.cuda()

            # CutMix
            img_x, mask_x = img_x1.clone(), mask_x1.clone()
            mask_x[cutmix_box == 1] = mask_x2[cutmix_box == 1]
            cutmix_box = cutmix_box.unsqueeze(1).repeat(1, 3, 1, 1)
            img_x[cutmix_box == 1] = img_x2[cutmix_box == 1]

            # Prediction
            model.train()
            pred_x = model(img_x)

            if args.nclass == 1:
                pred_x = pred_x.squeeze(1)
                visualise_pred = (pred_x > cfg['output_thresh']).int()
            else:
                conf_x = pred_x.softmax(dim=1).max(dim=1)[0]
                visualise_pred = pred_x.argmax(dim=1) * (conf_x > cfg['output_thresh']).int()

            # Visualising 
            if epoch > 0 and (epoch+1) % args.epochs_before_eval == 0:
                epoch_itr = (i % num_labeled_batches) * cfg['batch_size']
                if (epoch_itr >= 8 and epoch_itr < 16):
                    visualise_eval(
                        img_x.clone(), mask_x.clone(), visualise_pred.clone(),
                        epoch_itr + 100, epoch, args, cfg
                    )

            # Loss
            if args.nclass == 1:
                mask_x = mask_x.float()

            loss = loss_fn(pred_x, mask_x)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.update(loss.item())

            # LR scheduler step
            iters = epoch * len(trainloader_l1) + i
            if cfg['scheduler'] == 'poly':
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                
                if cfg['pretrained']:
                    optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            # Eval
            is_eval_epoch = (epoch > 0) and (epoch % args.epochs_before_eval == 0)

            if not is_eval_epoch or \
                (i % num_labeled_batches != 0):
                continue

            print(f"----- Eval [{epoch // args.epochs_before_eval}/{args.num_epochs // args.epochs_before_eval}]\t loss_all: {total_loss.val}")

            log({
                'epoch': epoch,
                'train_epoch/loss_all': total_loss.avg,
                'train_epoch/loss_all_var': total_loss.var,
            })

            total_loss.reset()

            eval_logs = evaluate(model, valloader, loss_fn, epoch, args, cfg)

            eval_logs['epoch_train/loss'] = total_loss.avg

            # Grand loss computation
            loss_t = eval_logs['epoch_train/loss']
            loss_v = eval_logs['eval/loss']
            wIoU = eval_logs['eval/wIoU']
            gl_weights = np.array(cfg['grand_loss_weights'])
            gl_losses = np.array([loss_t, loss_v, 1 - wIoU])
            
            # gl_losses = np.log(gl_losses)
            grand_loss = sum(gl_weights * gl_losses / sum(gl_weights))

            is_locally_best = wIoU > locally_best_iou
            locally_best_iou = max(wIoU, locally_best_iou)

            eval_logs['main/wIoU'] = locally_best_iou
            eval_logs['main/grand_loss'] = grand_loss

            log({
                'main/wIoU': locally_best_iou,
                'main/grand_loss': grand_loss
            })

            is_globally_best = wIoU > globally_best_iou
            globally_best_iou = max(wIoU, globally_best_iou)

            checkpoint_data = {
                'cfg': cfg,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            # if epoch > 10 and is_locally_best:
            if is_locally_best:
                checkpoint_data['locally_best_iou'] = locally_best_iou
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    checkpoint_path = Path(checkpoint_dir) / "locally_best.pth"
                    torch.save(checkpoint_data, checkpoint_path)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    if ray_train is not None:
                        ray_train.report(eval_logs, checkpoint=checkpoint)
            else:
                if ray_train is not None:
                    ray_train.report(eval_logs)

            # if epoch > 10 and is_globally_best:
            if is_globally_best:
                checkpoint_data['globally_best_iou'] = globally_best_iou
                torch.save(checkpoint_data, os.path.join(args.save_path, 'globally_best.pth'))


    print("Training Completed! Getting Test Results...")
    
    checkpoint_path = os.path.join(args.save_path, 'globally_best.pth')
    test_fig_dir = os.path.join(args.save_path, 'fig')
    test_npy_dir = os.path.join(args.save_path, 'npy')

    generate_test_outputs(
        checkpoint_path=checkpoint_path,
        test_fig_dir=test_fig_dir,
        test_npy_dir=test_npy_dir,
        args=args, cfg=cfg
    )

    print('All Test Results Generated!')

def main():
    args = get_args()
    print(vars(args))

    print("="*20)
    os.makedirs(args.save_path, exist_ok=True)

    config = {
        'grand_loss_weights': [1.0, 2.0, 4.0],

        'crop_size': 800,
        'batch_size': 2,  # 2, 4, 8, 16

        'backbone': 'efficientnet-b0',
        'pretrained': True,  # False, True

        'loss_fn': 'cross_entropy',  # 'cross_entropy', 'jaccard', 'combined'
        'lr': 2e-4,

        'lr_multi': 10.0,  # used only when pretrained is true
        'weight_decay': 1e-9,

        'scheduler': 'poly',

        'data_normalization': 'labeled',  # 'none', 'labeled', 'validation', 'unlabeled'

        'output_thresh' : 0.5,  # 0.5, 0.7, 0.9

        'p_jitter_l': 0.25,
        'p_gray_l'  : 0.0,
        'p_blur_l'  : 0.25,
        'p_cutmix_l': 0.0,
    }

    trainer(None, args, config)

    print("="*20)


if __name__ == "__main__":
    main()