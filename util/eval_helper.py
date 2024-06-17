import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from util.util import *


def intersectionAndUnion(pred, target, args, cfg):
    nclass = args.nclass

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # 'K' classes, pred and target sizes are N or N * L or 
    # N * H * W, each value in range 0 to K - 1.
    assert pred.ndim in [1, 2, 3]
    assert pred.shape == target.shape
    pred = pred.reshape(pred.size).copy()
    target = target.reshape(target.size)
    intersection = pred[np.where(pred == target)[0]]

    if args.nclass == 1:
        pred = pred.astype(np.uint8)
        intersection = intersection.astype(np.uint8)

        area_intersection = np.logical_and(pred, target).sum()
        area_union = np.logical_or(pred, target).sum()

        return area_intersection, area_union

    area_intersection, _ = np.histogram(intersection, bins=np.arange(nclass + 1))
    area_pred, _ = np.histogram(pred, bins=np.arange(nclass + 1))
    area_target, _ = np.histogram(target, bins=np.arange(nclass + 1))
    area_union = area_pred + area_target - area_intersection
    return area_intersection, area_union


def get_eval_scores(intersection, union, args, cfg, smooth=1e-10):
    iou_class = (intersection.sum + smooth) / (union.sum + smooth)
    mIoU = np.mean(iou_class)

    if args.nclass == 1:
        return {
            'eval/wIoU': mIoU
        }

    mean_i = intersection.avg
    mean_u = union.avg

    var_i = intersection.var
    var_u = union.var

    iou_class_var = (mean_i / (mean_u + smooth))**2

    tmp1 = (var_i + smooth / (mean_i + smooth))**2
    tmp2 = (var_u + smooth / (mean_u + smooth))**2
    iou_class_var = iou_class_var * (tmp1 + tmp2)

    # mIoU_var = np.mean(iou_class_var)

    class_weights = np.array(cfg['class_weights'])
    iou_weights = class_weights / sum(class_weights)
    wIoU = sum(iou_class * iou_weights)

    # wIoU_var = sum(iou_class_var * iou_weights)

    scores = {
        'eval/wIoU': wIoU,
        # 'eval/wIoU_var': wIoU_var,
    }

    for i, iou in enumerate(iou_class):
        scores[f'eval/idx_{i}_iou'] = iou

    return scores


def visualise_eval(img, target, pred, idx, epoch, args, cfg):
    if isinstance(idx, torch.Tensor):
        idx = idx.item()
    
    img_np = img.detach().cpu().numpy()

    if args.nclass > 1:
        target = F.one_hot(target, args.nclass)
        pred = F.one_hot(pred, args.nclass)

    target_np = target.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    for i in range(img_np.shape[0]):
        img = img_np[i].transpose(1,2,0)
        target = 255 * target_np[i]
        pred = 255 * pred_np[i]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Val Epoch {epoch}')

        img -= img.min()
        img /= img.max()
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(target, cmap='gray')
        axs[1].set_title('True Mask')
        axs[1].axis('off')

        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        if args.enable_logging:
            wandb.log({f"ValImages/idx_{idx}": wandb.Image(fig)}, commit=False)
        
        plt.close(fig)

    
def visualise_test(img, pred, save_path, args, cfg):
    img_np = img.detach().cpu().numpy()

    if args.nclass > 1:
        pred = F.one_hot(pred, args.nclass)

    pred_np = pred.detach().cpu().numpy()

    for i in range(img_np.shape[0]):
        img = img_np[i].transpose(1,2,0)
        pred = 255 * pred_np[i]

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{save_path}')

        img -= img.min()
        img /= img.max()
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(pred, cmap='gray')
        axs[1].set_title('Predicted Mask')
        axs[1].axis('off')

        # if args.enable_logging:
        #     wandb.log({f"TestImages/{filename}": wandb.Image(fig)}, commit=False)

        plt.savefig(save_path)
        plt.close(fig)


def visualise_train(
        img_x, mask_x, pred_x,
        img_u_s, mask_u_w_cutmixed, pred_u_s,
        idx, epoch, args, cfg
    ):
    
    if args.nclass == 1:
        pred_x = pred_x.sigmoid() > cfg['conf_thresh']
        pred_u_s = pred_u_s.sigmoid() > cfg['conf_thresh']
    else:
        # pred_x = (pred_x.softmax(dim=1) > cfg['conf_thresh']).int()
        pred_x = pred_x.argmax(dim=1)

        # pred_u_s = (pred_u_s.softmax(dim=1) > cfg['conf_thresh']).int()
        pred_u_s = pred_u_s.argmax(dim=1)

    img_x_np = img_x.detach().cpu().numpy()
    img_u_s_np = img_u_s.detach().cpu().numpy()

    if args.nclass > 1:
        mask_x = F.one_hot(mask_x, args.nclass)
        pred_x = F.one_hot(pred_x, args.nclass)
        mask_u_w_cutmixed = F.one_hot(mask_u_w_cutmixed, args.nclass)
        pred_u_s = F.one_hot(pred_u_s, args.nclass)

    mask_x_np = mask_x.detach().cpu().numpy()
    pred_x_np = pred_x.detach().cpu().numpy()
    mask_u_w_cutmixed_np = mask_u_w_cutmixed.detach().cpu().numpy()
    pred_u_s_np = pred_u_s.detach().cpu().numpy()

    for i in range(img_x_np.shape[0]):
        img_x = img_x_np[i].transpose(1,2,0)
        img_u_s = img_u_s_np[i].transpose(1,2,0)

        mask_x = 255 * mask_x_np[i]
        pred_x = 255 * pred_x_np[i]
        mask_u_w_cutmixed = 255 * mask_u_w_cutmixed_np[i]
        pred_u_s = 255 * pred_u_s_np[i]

        fig, axs = plt.subplots(2, 3, figsize=(15,15))
        fig.suptitle(f'Train Epoch {epoch}')

        # Labeled
        img_x -= img_x.min()
        img_x /= img_x.max()
        axs[0,0].imshow(img_x)
        axs[0,0].set_title('img_x')
        axs[0,0].axis('off')

        axs[0,1].imshow(mask_x)
        axs[0,1].set_title('mask_x')
        axs[0,1].axis('off')

        axs[0,2].imshow(pred_x)
        axs[0,2].set_title('pred_x')
        axs[0,2].axis('off')

        # Unlabeled Strong Augmentation
        img_u_s -= img_u_s.min()
        img_u_s /= img_u_s.max()
        axs[1,0].imshow(img_u_s)
        axs[1,0].set_title('img_u_s')
        axs[1,0].axis('off')

        axs[1,1].imshow(mask_u_w_cutmixed, cmap='gray')
        axs[1,1].set_title('mask_u_w_cutmixed')
        axs[1,1].axis('off')

        axs[1,2].imshow(pred_u_s, cmap='gray')
        axs[1,2].set_title('pred_u_s')
        axs[1,2].axis('off')

        if args.enable_logging:
            wandb.log({f"TrainImages/idx_{idx}_{i}": wandb.Image(fig)}, commit=False)
        
        plt.close(fig)