import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def dice_coefficient(tp, fp, fn, tn, smooth=1e-6):
    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    return ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item()


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
    area_intersection, _ = np.histogram(intersection, bins=np.arange(nclass + 1))
    area_pred, _ = np.histogram(pred, bins=np.arange(nclass + 1))
    area_target, _ = np.histogram(target, bins=np.arange(nclass + 1))
    area_union = area_pred + area_target - area_intersection
    return area_intersection, area_union


def get_eval_scores(intersection, union, cfg, smooth=1e-10):
    iou_class = intersection.sum / (union.sum + smooth)
    # mIoU = np.mean(iou_class)

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
    img_np = img.detach().cpu().numpy()
    target_np = F.one_hot(target, args.nclass).detach().cpu().numpy()
    pred_np = F.one_hot(pred, args.nclass).detach().cpu().numpy()

    for i in range(img_np.shape[0]):
        img = img_np[i].transpose(1,2,0)
        target = 255 * target_np[i]
        pred = 255 * pred_np[i]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Val Epoch {epoch}')
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
            wandb.log({f"ValImages/idx_{idx.item()}": wandb.Image(fig)}, commit=False)
        
        plt.close(fig)


def visualise_train(
        img_x, mask_x, pred_x,
        img_u_w, img_u_s, 
        pred_u_w, pred_u_s,
        idx, epoch, args, cfg
    ):
    
    if args.nclass == 1:
        pred_x = pred_x.sigmoid() > cfg['conf_thresh']
        pred_u_w = pred_u_w.sigmoid() > cfg['conf_thresh']
        pred_u_s = pred_u_s.sigmoid() > cfg['conf_thresh']
    else:
        pred_x = (pred_x.softmax(dim=1) > cfg['conf_thresh']).int()
        pred_x = pred_x.argmax(dim=1)

        pred_u_w = (pred_u_w.softmax(dim=1) > cfg['conf_thresh']).int()
        pred_u_w = pred_u_w.argmax(dim=1)

        pred_u_s = (pred_u_s.softmax(dim=1) > cfg['conf_thresh']).int()
        pred_u_s = pred_u_s.argmax(dim=1)

    img_x_np = img_x.detach().cpu().numpy()
    img_u_w_np = img_u_w.detach().cpu().numpy()
    img_u_s_np = img_u_s.detach().cpu().numpy()

    mask_x_np = F.one_hot(mask_x, args.nclass).detach().cpu().numpy()
    pred_x_np = F.one_hot(pred_x, args.nclass).detach().cpu().numpy()
    pred_u_w_np = F.one_hot(pred_u_w, args.nclass).detach().cpu().numpy()
    pred_u_s_np = F.one_hot(pred_u_s, args.nclass).detach().cpu().numpy()

    for i in range(img_u_w_np.shape[0]):
        img_x = img_x_np[i].transpose(1,2,0)
        img_u_w = img_u_w_np[i].transpose(1,2,0)
        img_u_s = img_u_s_np[i].transpose(1,2,0)

        mask_x = 255 * mask_x_np[i]
        pred_x = 255 * pred_x_np[i]
        pred_u_w = 255 * pred_u_w_np[i]
        pred_u_s = 255 * pred_u_s_np[i]

        fig, axs = plt.subplots(3, 3, figsize=(15,15))
        fig.suptitle(f'Train Epoch {epoch}')

        # Labeled
        axs[0,0].imshow(img_x)
        axs[0,0].set_title('img_x')
        axs[0,0].axis('off')

        axs[0,1].imshow(mask_x)
        axs[0,1].set_title('mask_x')
        axs[0,1].axis('off')

        axs[0,2].imshow(pred_x)
        axs[0,2].set_title('pred_x')
        axs[0,2].axis('off')
        
        # Unlabeled Weak Augmentation
        axs[1,0].imshow(img_u_w)
        axs[1,0].set_title('img_u_w')
        axs[1,0].axis('off')

        axs[1,1].imshow(pred_u_w, cmap='gray')
        axs[1,1].set_title('pred_u_w')
        axs[1,1].axis('off')
        axs[1,2].axis('off')

        # Unlabeled Strong Augmentation
        axs[2,0].imshow(img_u_s)
        axs[2,0].set_title('img_u_s')
        axs[2,0].axis('off')

        axs[2,1].imshow(pred_u_s, cmap='gray')
        axs[2,1].set_title('pred_u_s')
        axs[2,1].axis('off')
        axs[2,2].axis('off')

        if args.enable_logging:
            wandb.log({f"TrainImages/idx_{idx}": wandb.Image(fig)}, commit=False)
        
        plt.close(fig)