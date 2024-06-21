import wandb
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from util.util import *
from util.train_helper import init_model

# Data
from torch.utils.data import DataLoader
from dataset.kerogens import KerogensDataset


def evaluate(
    model, valloader, criterion,
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

            val_loss = criterion(pred, mask)
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
        test_fig_dir, test_npy_dir,
        args, cfg
    ):
    
    os.makedirs(test_fig_dir, exist_ok=True)
    os.makedirs(test_npy_dir, exist_ok=True)

    testset = KerogensDataset(
        args.test_data_dir, 'test',
        args, cfg
    )

    testloader = DataLoader(
        testset, batch_size=1, pin_memory=True, 
        num_workers=1, drop_last=False
    )

    model = init_model(args, cfg, checkpoint_path)

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
            visualise_test(raw_img, pred, filename, test_fig_dir, args, cfg)

            pred_np = pred.detach().cpu().numpy()
            np.save(os.path.join(test_npy_dir, filename), pred_np)



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

    
def visualise_test(img, pred, filename, save_dir, args, cfg, show_now=False):
    save_path = os.path.join(save_dir, filename)
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

        if args.enable_logging:
            wandb.log({f"TestImages/{filename}": wandb.Image(fig)})

        if show_now:
            plt.show()
        else:
            plt.savefig(save_path)

        plt.close(fig)


def visualise_train(
        idx, epoch, args, cfg,
        img_x, mask_x, pred_x,
        img_u_s1, mask_u_w_cutmixed1, pred_u_s1,
        img_u_s2=None, mask_u_w_cutmixed2=None, pred_u_s2=None,
    ):

    is_unimatch = img_u_s2 != None
    
    if args.nclass == 1:
        pred_x = pred_x.sigmoid() > cfg['output_thresh']
        pred_u_s1 = pred_u_s1.sigmoid() > cfg['output_thresh']

        if is_unimatch:
            pred_u_s2 = pred_u_s2.sigmoid() > cfg['output_thresh']
    else:
        conf_x = pred_x.softmax(dim=1).max(dim=1)[0]
        pred_x = pred_x.argmax(dim=1) * \
            (conf_x > cfg['output_thresh']).int()
        
        conf_u_s1 = pred_u_s1.softmax(dim=1).max(dim=1)[0]
        pred_u_s1 = pred_u_s1.argmax(dim=1) * \
            (conf_u_s1 > cfg['output_thresh']).int()
        
        if is_unimatch:
            conf_u_s2 = pred_u_s2.softmax(dim=1).max(dim=1)[0]
            pred_u_s2 = pred_u_s2.argmax(dim=1) * \
                (conf_u_s2 > cfg['output_thresh']).int()
                        
    img_x_np = img_x.detach().cpu().numpy()
    img_u_s1_np = img_u_s1.detach().cpu().numpy()

    if is_unimatch:
        img_u_s2_np = img_u_s2.detach().cpu().numpy()

    if args.nclass > 1:
        mask_x = F.one_hot(mask_x, args.nclass)
        pred_x = F.one_hot(pred_x, args.nclass)

        mask_u_w_cutmixed1 = F.one_hot(mask_u_w_cutmixed1, args.nclass)

        if is_unimatch:
            mask_u_w_cutmixed2 = F.one_hot(mask_u_w_cutmixed2, args.nclass)
        
        pred_u_s1 = F.one_hot(pred_u_s1, args.nclass)

        if is_unimatch:
            pred_u_s2 = F.one_hot(pred_u_s2, args.nclass)


    mask_x_np = mask_x.detach().cpu().numpy()
    pred_x_np = pred_x.detach().cpu().numpy()
    mask_u_w_cutmixed1_np = mask_u_w_cutmixed1.detach().cpu().numpy()
    pred_u_s1_np = pred_u_s1.detach().cpu().numpy()

    if is_unimatch:
        mask_u_w_cutmixed2_np = mask_u_w_cutmixed2.detach().cpu().numpy()
        pred_u_s2_np = pred_u_s2.detach().cpu().numpy()

    for i in range(img_x_np.shape[0]):
        img_x = img_x_np[i].transpose(1,2,0)
        img_u_s1 = img_u_s1_np[i].transpose(1,2,0)
        
        if is_unimatch:
            img_u_s2 = img_u_s2_np[i].transpose(1,2,0)

        mask_x = 255 * mask_x_np[i]
        pred_x = 255 * pred_x_np[i]
        mask_u_w_cutmixed1 = 255 * mask_u_w_cutmixed1_np[i]
        pred_u_s1 = 255 * pred_u_s1_np[i]

        if is_unimatch:
            mask_u_w_cutmixed2 = 255 * mask_u_w_cutmixed2_np[i]
            pred_u_s2 = 255 * pred_u_s2_np[i]

        if is_unimatch:
            fig, axs = plt.subplots(3, 3, figsize=(15,15))
        else:
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

        # Unlabeled Strong Augmentation 1
        img_u_s1 -= img_u_s1.min()
        img_u_s1 /= img_u_s1.max()
        axs[1,0].imshow(img_u_s1)
        axs[1,0].set_title('img_u_s1')
        axs[1,0].axis('off')

        axs[1,1].imshow(mask_u_w_cutmixed1, cmap='gray')
        axs[1,1].set_title('mask_u_w_cutmixed1')
        axs[1,1].axis('off')

        axs[1,2].imshow(pred_u_s1, cmap='gray')
        axs[1,2].set_title('pred_u_s1')
        axs[1,2].axis('off')

        if is_unimatch:
            # Unlabeled Strong Augmentation 2
            img_u_s2 -= img_u_s2.min()
            img_u_s2 /= img_u_s2.max()
            axs[2,0].imshow(img_u_s2)
            axs[2,0].set_title('img_u_s2')
            axs[2,0].axis('off')

            axs[2,1].imshow(mask_u_w_cutmixed2, cmap='gray')
            axs[2,1].set_title('mask_u_w_cutmixed2')
            axs[2,1].axis('off')

            axs[2,2].imshow(pred_u_s2, cmap='gray')
            axs[2,2].set_title('pred_u_s2')
            axs[2,2].axis('off')

        if args.enable_logging:
            wandb.log({f"TrainImages/idx_{idx}_{i}": wandb.Image(fig)}, commit=False)
        
        plt.close(fig)