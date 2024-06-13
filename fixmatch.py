# General
import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data
from torch.utils.data import DataLoader
from dataset.kerogens import KerogensDataset

# smp
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss

# utils
from util.util import *
from util.train_helper import *
from util.eval_helper import *

# Set the random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(args, cfg):
    # Datasets
    label_count = len(os.listdir(os.path.join(args.labeled_data_dir, 'label')))
    num_unlabeled = int(cfg['unlabeled_ratio'] * label_count)
    trainset_u = KerogensDataset(
        args.unlabeled_data_dir, 
        'train_u', 
        args, cfg,
        cfg['crop_size'], num_unlabeled)
    trainset_l = KerogensDataset(
        args.labeled_data_dir, 
        'train_l', 
        args, cfg,
        cfg['crop_size'], num_unlabeled)
    valset = KerogensDataset(
        args.val_data_dir, 'val',
        args, cfg)

    # Dataloaders
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'],
        pin_memory=True, num_workers=1, drop_last=True)

    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'],
        pin_memory=True, num_workers=1, drop_last=True)

    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1,
        drop_last=False)
    
    return trainloader_l, trainloader_u, valloader


def run_epoch(
        model, optimizer,
        criterion_l, criterion_u, 
        trainloader_l, trainloader_u, valloader,
        epoch, total_iters,
        args, cfg
    ):

    loader = zip(trainloader_l, trainloader_u)

    total_loss  = AverageMeter(track_variance=True)
    total_loss_x = AverageMeter(track_variance=True)
    total_loss_s = AverageMeter(track_variance=True)
    total_mask_ratio = AverageMeter()

    model.train()

    for i, ((img_x, mask_x),
            (img_u_w, img_u_s, _, _, _)) in enumerate(loader):

        img_x, mask_x = img_x.cuda(), mask_x.cuda()
        img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()

        num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

        pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])
        pred_u_s = model(img_u_s)

        if i < 9:
            visualise_train(
                img_x.clone(), mask_x.clone(), pred_x.clone(),
                img_u_w.clone(), img_u_s.clone(), 
                pred_u_w.clone(), pred_u_s.clone(),
                i, epoch, args, cfg
            )

        if args.nclass == 1:
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.sigmoid()
            mask_u_w = pred_u_w > cfg['conf_thresh']

            pred_x = pred_x.squeeze(1)
        else:
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

        loss_x = criterion_l(pred_x, mask_x)
        
        loss_u_s = criterion_u(pred_u_s, mask_u_w)
        loss_u_s = loss_u_s * (conf_u_w > cfg['conf_thresh'])
        loss_u_s = loss_u_s.mean()

        loss = (loss_x + loss_u_s) / 2.0            

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # LR scheduler step
        iters = epoch * len(trainloader_l) + i
        if cfg['scheduler'] == 'poly':
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']


        total_loss.update(loss.item())
        total_loss_x.update(loss_x.item())
        total_loss_s.update(loss_u_s.item())

        mask_ratio = (conf_u_w > cfg['conf_thresh']).float().mean().item()
        total_mask_ratio.update(mask_ratio)
        
        # log({
        #     'iters': iters,
        #     'train_batch/loss_all': total_loss.val,
        #     'train_batch/loss_x': total_loss_x.val,
        #     'train_batch/loss_s': total_loss_s.val,
        #     'train_batch/mask_ratio': total_mask_ratio.val,
        # })

        if i % (len(trainloader_u) // 8) == 0:
            print(f"\tIter [{i}/{len(trainloader_u)}]\t loss_all: {total_loss.val}")

        # debug
        # break

    log({
        'epoch': epoch,
        'train_epoch/loss_all': total_loss.avg,
        'train_epoch/loss_all_var': total_loss.var,

        'train_epoch/loss_x': total_loss_x.avg,
        'train_epoch/loss_x_var': total_loss_x.var,
        
        'train_epoch/loss_s': total_loss_s.avg,
        # 'train_epoch/loss_s_var': total_loss_s.var,

        'train_epoch/mask_ratio': total_mask_ratio.avg,
    })


    model.eval()

    total_val_loss = AverageMeter(track_variance=True)
    intersection_meter = AverageMeter(track_variance=True)
    union_meter = AverageMeter(track_variance=True)
    
    with torch.no_grad():
        for idx, img, mask in valloader:
            raw_img, mask = img.cuda(), mask.cuda()

            h, w = raw_img.shape[-2:]
            img = F.interpolate(raw_img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

            pred = model(img)
            
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)

            val_loss = criterion_l(pred, mask)
            total_val_loss.update(val_loss)

            if args.nclass == 1:
                pred = pred.sigmoid() > cfg['conf_thresh']
            else:
                # pred = (pred.softmax(dim=1) > cfg['conf_thresh']).int()
                pred = pred.argmax(dim=1)  # TODO: check if this works

            intersection, union = intersectionAndUnion(pred, mask, args, cfg)

            visualise_eval(raw_img, mask, pred, idx, epoch, args, cfg)

            intersection_meter.update(intersection)
            union_meter.update(union)

            # debug
            # break

    eval_scores = get_eval_scores(intersection_meter, union_meter, cfg)
    eval_scores['eval/loss'] = total_val_loss.avg
    eval_scores['eval/loss_var'] = total_val_loss.var

    log(eval_scores)

    wIoU = eval_scores['eval/wIoU']
    print(f"----- wIoU: {wIoU}\t Loss: {total_val_loss.avg}")

    return {
        'epoch_train/loss': total_loss.avg,
        'eval/loss': total_val_loss.avg.item(),
        'eval/wIoU': wIoU
    } 


def trainer(args, cfg):
    init_logging(args, cfg)

    model = init_model(args.nclass, cfg['backbone'])
    print(f"Param count: {count_params(model):.1f}M")
    optimizer = init_optimizer(model, cfg)

    class_weights = torch.tensor(cfg['class_weights']).float().cuda()

    if args.nclass == 1:
        # criterion_jaccard = JaccardLoss("binary")
        pass
    else:
        # criterion_jaccard = JaccardLoss("multiclass")
        
        # criterion_l = criterion_jaccard
        # criterion_u = criterion_jaccard

        criterion_l = nn.CrossEntropyLoss(class_weights)
        criterion_u = nn.CrossEntropyLoss(class_weights, reduction='none')


    trainloader_l, trainloader_u, valloader = load_data(args, cfg)

    total_iters = len(trainloader_l) * args.num_epochs

    best_iou = 0
    epoch = -1

    if args.use_checkpoint and os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']

        print(f"Loading checkpoint from epoch: {epoch}")

    print("Starting Training...")
    for epoch in range(epoch + 1, args.num_epochs):
        print(f"Epoch [{epoch}/{args.num_epochs}]\t Previous Best IoU: {best_iou}")

        logs = run_epoch(
            model, optimizer,
            criterion_l, criterion_u, 
            trainloader_l, trainloader_u, valloader,
            epoch, total_iters,
            args, cfg
        )

        wIoU = logs['eval/wIoU']
        is_best = wIoU > best_iou
        best_iou = max(wIoU, best_iou)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_iou': best_iou,
        }

        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

    print("Training Completed!")


def main():
    set_seed(42)

    args = get_args()

    print("="*20)
    os.makedirs(args.save_path, exist_ok=True)

    config = {
        'crop_size': 800,
        'batch_size': 2, 
        'unlabeled_ratio': 10,

        'backbone': 'efficientnet-b0',
        
        'class_weights': np.array([0.008, 1.0, 0.048]),
        'lr': 1e-4,
        'lr_multi': 10.0,
        'weight_decay': 1e-9,
        'scheduler': 'poly',

        'conf_thresh': 0.78,
        'p_jitter': 0.3186,
        'p_gray': 0.6534,
        'p_blur': 0.2515,
    }

    trainer(args, config)
    print("="*20)


if __name__ == "__main__":
    main()