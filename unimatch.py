# General
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
    trainset_u = KerogensDataset(
        args.unlabeled_data_dir, 'train_u', 
        args, cfg,
        cfg['crop_size'], nsample
    )
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
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'],
        pin_memory=True, num_workers=1, drop_last=True
    )

    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'],
        pin_memory=True, num_workers=1, drop_last=True
    )

    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1,
        drop_last=False
    )
    
    return trainloader_l, trainloader_u, valloader


def evaluate(
    model, valloader, criterion_l,
    epoch,
    args, cfg
):
    model.eval()

    total_val_loss = AverageMeter(track_variance=True)
    intersection_meter = AverageMeter(track_variance=True)
    union_meter = AverageMeter(track_variance=True)
    
    with torch.no_grad():
        for idx, img, mask in valloader:
            if idx > 2 and args.fast_debug:
                break

            raw_img, mask = img.cuda(), mask.cuda()

            h, w = raw_img.shape[-2:]
            img = F.interpolate(raw_img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

            pred = model(img)
            
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)

            if args.nclass == 1:
                pred = pred.squeeze(1)
                mask = mask.float()

            val_loss = criterion_l(pred, mask)
            total_val_loss.update(val_loss)

            if args.nclass == 1:
                pred = (pred.sigmoid() > cfg['output_thresh']).int()
            else:
                conf = pred.softmax(dim=1).max(dim=1)[0]
                pred = pred.argmax(dim=1) * (conf > cfg['output_thresh']).int()

            intersection, union = intersectionAndUnion(pred, mask, args, cfg)

            visualise_eval(raw_img, mask, pred, idx, epoch, args, cfg)

            intersection_meter.update(intersection)
            union_meter.update(union)

    eval_scores = get_eval_scores(intersection_meter, union_meter, args, cfg)
    eval_scores['eval/loss'] = total_val_loss.avg
    eval_scores['eval/loss_var'] = total_val_loss.var

    log(eval_scores)

    wIoU = eval_scores['eval/wIoU']
    print(f"----- wIoU: {wIoU}\t Loss: {total_val_loss.avg}")

    return {
        'eval/loss': total_val_loss.avg.item(),
        'eval/wIoU': wIoU
    } 


def generate_test_outputs(
        checkpoint_path, 
        output_dir, 
        args, cfg
    ):
    submission_dir = 'outputs/submission/'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    testset = KerogensDataset(
        args.test_data_dir, 'test',
        args, cfg
    )

    testloader = DataLoader(
        testset, batch_size=1, pin_memory=True, 
        num_workers=1, drop_last=False
    )

    model = init_model(args.nclass, cfg['backbone'], checkpoint_path)

    model.eval()
    with torch.no_grad():
        for filename, img in testloader:
            print(filename[0])
            raw_img = img.cuda()

            h, w = raw_img.shape[-2:]
            img = F.interpolate(raw_img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

            pred = model(img)
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)

            if args.nclass == 1:
                pred = pred.squeeze(1)
                pred = (pred.sigmoid() > cfg['output_thresh']).int() * 3  # since idx_3
            else:
                conf = pred.softmax(dim=1).max(dim=1)[0]
                pred = pred.argmax(dim=1) * (conf > cfg['output_thresh']).int()

            filename = filename[0] + "_pred"
            visualise_test(raw_img, pred, os.path.join(output_dir, filename), args, cfg)

            pred_np = pred.detach().cpu().numpy()
            np.save(os.path.join(submission_dir, filename), pred_np)


def trainer(ray_train, args, cfg):
    assert cfg['unlabeled_ratio'] % args.epochs_before_eval == 0, \
        "The chosen `unlabeled_ratio` is not a multiple of `epochs_before_eval`"
    
    global globally_best_iou

    init_logging(args, cfg)

    model = init_model(args.nclass, cfg['backbone'])
    print(f"Param count: {count_params(model):.1f}M")
    optimizer = init_optimizer(model, cfg)

    if args.nclass > 1:
        cw_list = np.array(cfg['class_weights'])
        cw_list[2] = cfg['class_weights_idx_2']
        class_weights = torch.tensor(cw_list).float().cuda()

    if args.nclass == 1:
        if cfg['loss_fn'] == 'cross_entropy':
            criterion_l = nn.BCEWithLogitsLoss()
            criterion_u = nn.BCEWithLogitsLoss()
        elif cfg['loss_fn'] == 'jaccard':
            criterion_l = smp.losses.JaccardLoss('binary')
            criterion_u = smp.losses.JaccardLoss('binary')
        elif cfg['loss_fn'] == 'dice':
            criterion_l = smp.losses.DiceLoss('binary')
            criterion_u = smp.losses.DiceLoss('binary')
    else:
        if cfg['loss_fn'] == 'cross_entropy':
            criterion_l = nn.CrossEntropyLoss(class_weights)
            criterion_u = nn.CrossEntropyLoss(class_weights)
        elif cfg['loss_fn'] == 'jaccard':
            criterion_l = smp.losses.JaccardLoss('multiclass')
            criterion_u = smp.losses.JaccardLoss('multiclass')
        elif cfg['loss_fn'] == 'dice':
            criterion_l = smp.losses.DiceLoss('multiclass')
            criterion_u = smp.losses.DiceLoss('multiclass')


    num_labeled = len(os.listdir(os.path.join(args.labeled_data_dir, 'label')))
    num_labeled_batches = num_labeled // cfg['batch_size']
    num_unlabeled = int(cfg['unlabeled_ratio'] * num_labeled)

    trainloader_l, trainloader_u, valloader = load_data(args, cfg, nsample=num_unlabeled)

    total_iters = len(trainloader_l) * args.num_epochs

    locally_best_iou = 0
    epoch = -1

    total_loss  = AverageMeter(track_variance=True)
    total_loss_x = AverageMeter(track_variance=True)
    total_loss_s = AverageMeter(track_variance=True)
    total_mask_ratio = AverageMeter()

    print("Starting Training...")
    while epoch < args.num_epochs:     
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):

            if i > 2 and args.fast_debug:
                break

            if epoch + 1 >= args.num_epochs:
                epoch += 1
                break

            if i % num_labeled_batches == 0:
                epoch += 1

                print(f"Epoch [{epoch}/{args.num_epochs}]\t Previous Best IoU: {locally_best_iou}")

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix, img_u_s1_mix, img_u_s2_mix = img_u_w_mix.cuda(), img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            # predicting inside cutmix box
            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()

                if args.nclass == 1:
                    pred_u_w_mix = pred_u_w_mix.squeeze(1)
                    mask_u_w_mix = (pred_u_w_mix.sigmoid() > cfg['output_thresh']).int()
                else:
                    conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                    mask_u_w_mix = pred_u_w_mix.argmax(dim=1) * \
                        (conf_u_w_mix > cfg['output_thresh']).int()

            # CutMix
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            # Prediction
            pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).chunk(2)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            if args.nclass == 1:
                pred_u_w = pred_u_w.squeeze(1)
                conf_u_w = pred_u_w.sigmoid()
                mask_u_w = (conf_u_w > cfg['output_thresh']).int()

                pred_x = pred_x.squeeze(1)
                pred_u_s1 = pred_u_s1.squeeze(1)
                pred_u_s2 = pred_u_s2.squeeze(1)
            else:
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1) * \
                        (conf_u_w > cfg['output_thresh']).int()

            # CutMix
            mask_u_w_cutmixed1 = mask_u_w.clone()
            mask_u_w_cutmixed2 = mask_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]

            # Visualising 
            if epoch > 0 and (epoch+1) % args.epochs_before_eval == 0:
                epoch_itr = (i % num_labeled_batches) * cfg['batch_size']
                if (epoch_itr >= 8 and epoch_itr < 16):
                    visualise_train(
                        epoch_itr, epoch, args, cfg,
                        img_x.clone(), mask_x.clone(), pred_x.clone(),
                        img_u_s1.clone(), mask_u_w_cutmixed1.clone(), pred_u_s1.clone(),
                        img_u_s2.clone(), mask_u_w_cutmixed2.clone(), pred_u_s2.clone(),
                    )

            if args.nclass == 1:
                mask_x = mask_x.float()
                mask_u_w_cutmixed = mask_u_w_cutmixed.float()

            # implement class weights for idx_12: jaccard and dice losses 
            loss_x = criterion_l(pred_x, mask_x)
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

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

            is_eval_epoch = (epoch > 0) and (epoch % args.epochs_before_eval == 0)

            if not is_eval_epoch or \
                (i % num_labeled_batches != 0):
                continue

            print(f"----- Eval [{epoch // args.epochs_before_eval}/{args.num_epochs // args.epochs_before_eval}]\t loss_all: {total_loss.val}")

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

            total_loss.reset()
            total_loss_x.reset()
            total_loss_s.reset()
            total_mask_ratio.reset()

            eval_logs = evaluate(model, valloader, criterion_l, epoch, args, cfg)

            eval_logs['epoch_train/loss'] = total_loss.avg

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

            # debug
            # if is_locally_best:
            if epoch > 10 and is_locally_best:
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

            # if is_globally_best:
            if epoch > 10 and is_globally_best:
                checkpoint_data['globally_best_iou'] = globally_best_iou
                torch.save(checkpoint_data, os.path.join(args.save_path, 'globally_best.pth'))


    print("Training Completed!")


def main():
    set_seed(42)

    args = get_args()

    print("="*20)
    os.makedirs(args.save_path, exist_ok=True)

    config = {
        'grand_loss_weights': [1.0, 2.0, 4.0],

        'crop_size': 800,
        'batch_size': 2, 
        'unlabeled_ratio': 10,

        'backbone': 'efficientnet-b0',
        
        'class_weights_idx_2': 0.05,
        'class_weights': [0.008, 1.0, 0.08],

        'loss_fn': 'cross_entropy',
        
        'lr': 5e-4,
        'lr_multi': 10.0,
        'weight_decay': 0.000003057,

        'scheduler': 'poly',

        'use_data_normalization': False,

        'conf_thresh': 0.6179,
        'p_jitter': 0.3778,
        'p_gray': 0.3867,
        'p_blur': 0.5215,
        'p_cutmix': 0.6652,

        'output_thresh': 0.6


        # 'use_data_normalization': True,

        # 'conf_thresh': 0.95,
        # 'p_jitter': 0.1,
        # 'p_gray': 0.4,
        # 'p_blur': 0.2,
        # 'p_cutmix': 0.5,

        # 'output_thresh': 0.6
    }

    generate_results = False

    if not generate_results:
        trainer(None, args, config)
    else:
        if args.dataset == 'idx_12':
            config['use_data_normalization'] = False
            config['output_thresh'] = 0.8
            generate_test_outputs(
                checkpoint_path="best_weights/idx_12/12a251be.pth",
                output_dir="outputs/idx_12/",
                args=args, cfg=config
            )
        elif args.dataset == 'idx_3':
            config['use_data_normalization'] = True
            config['output_thresh'] = 0.6
            generate_test_outputs(
                checkpoint_path="best_weights/idx_3/globally_best.pth",
                output_dir="outputs/idx_3/",
                args=args, cfg=config
            )

    print("="*20)


if __name__ == "__main__":
    main()