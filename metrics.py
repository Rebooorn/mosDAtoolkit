import torchmetrics.functional as tf
import torch
# from mostoolkit.vis_utils import slice_visualize_XY
import torch.nn.functional as tnf
import numpy as np
from mostoolkit.ct_utils import crop_to_standard


def calculate_metrics_ctorg(y_hat: list, y: list, max_shape,):
    # mean dice and organ-wise dice: bg, bladder, bone, liver, lung, kidney,
    # 
    # assert y_hat.shape == y.shape, 'pred and gt sh'
    yy_hat = []
    yy = []
    for p,gt in zip(y_hat, y):
        yy_hat.append(torch.Tensor(crop_to_standard(p, max_shape)))
        yy.append(torch.Tensor(crop_to_standard(gt, max_shape)))
    
    yy_hat = torch.stack(yy_hat, dim=0)
    yy = torch.stack(yy, dim=0)
    # print(max_shape)
    # print(yy_hat.shape, yy.shape)
    avg_dice_micro = tf.dice(yy_hat.long(), yy.long(),average='micro', ignore_index=0, num_classes=6)
    # avg_dice_macro = tf.dice(yy_hat.long(), yy.long(),average='macro', ignore_index=0, num_classes=6)
    class_dice = tf.dice(yy_hat.long(), yy.long(), average=None, num_classes=6).numpy()
    metrics = {
        f'avg_dice_micro': avg_dice_micro.item(),
        # f'avg_dice_macro': avg_dice_macro.item(),
        f'dice_liver': class_dice[1],
        f'dice_bladder': class_dice[2],
        f'dice_lung': class_dice[3],
        f'dice_kidney': class_dice[4],
        f'dice_bone': class_dice[5],
    }

    return metrics


def calculate_metrics_amos(y_hat: list, y: list, max_shape,):
    # mean dice and organ-wise dice: bg, bladder, bone, liver, lung, kidney,
    # 
    # y_hat = torch.argmax(y_hat, dim=1).squeeze(0)
    # y_hat = y_hat
    # y = y.squeeze(0).squeeze(0)
    # {"0": "background", "1": "spleen", "2": "right kidney", "3": "left kidney", "4": "gall bladder", "5": "esophagus", "6": "liver", "7": "stomach", 
    # "8": "arota", "9": "postcava", "10": "pancreas", "11": "right adrenal gland", "12": "left adrenal gland", "13": "duodenum", "14": "bladder", 
    # "15": "prostate/uterus"}
    # assert y_hat.shape == y.shape, 'pred and gt sh'

    # This does not work for test dataset, run it only on woody
    yy_hat = []
    yy = []
    for p,gt in zip(y_hat, y):
        yy_hat.append(torch.Tensor(crop_to_standard(p, max_shape)))
        yy.append(torch.Tensor(crop_to_standard(gt, max_shape)))
    
    yy_hat = torch.stack(yy_hat, dim=0)
    yy = torch.stack(yy, dim=0)
    # print(max_shape)
    # print(yy_hat.shape, yy.shape)
    avg_dice_micro = tf.dice(yy_hat.long(), yy.long(),average='micro', ignore_index=0, num_classes=16)
    avg_dice_macro = tf.dice(yy_hat.long(), yy.long(),average='macro', ignore_index=0, num_classes=16)
    class_dice = tf.dice(yy_hat.long(), yy.long(), average=None, num_classes=16).numpy()

    metrics = {
        f'avg_dice_micro': avg_dice_micro.item(),
        f'avg_dice_macro': avg_dice_macro.item(),
        f'dice_spleen': class_dice[1],
        f'dice_rkidney': class_dice[2],
        f'dice_lkidney': class_dice[3],
        f'dice_gbladder': class_dice[4],
        f'dice_esophagus': class_dice[5],
        f'dice_liver': class_dice[6],
        f'dice_stomach': class_dice[7],
        f'dice_aorta': class_dice[8],
        f'dice_postcava': class_dice[9],
        f'dice_pancreas': class_dice[10],
        f'dice_radrenalg': class_dice[11],
        f'dice_ladrenalg': class_dice[12],
        f'dice_duodenum': class_dice[13],
        f'dice_bladder': class_dice[14],
        f'dice_prostate': class_dice[15],
    }

    return metrics


def calculate_metrics_dect(y_hat: list, y: list, max_shape,):
    # mean dice and organ-wise dice: bg, bladder, bone, liver, lung, kidney,
    # 
    # y_hat = torch.argmax(y_hat, dim=1).squeeze(0)
    # y_hat = y_hat
    # y = y.squeeze(0).squeeze(0)
    # {'background': 0, 'lkidney': 1, 'rkidney': 2, 'liver': 3, 'spleen': 4, 'llung': 5, 'rlung': 6, 'pancreas': 7, 'gbladder': 8, 'aorta': 9, }
    # assert y_hat.shape == y.shape, 'pred and gt sh'
    yy_hat = []
    yy = []
    for p,gt in zip(y_hat, y):
        yy_hat.append(torch.Tensor(crop_to_standard(p, max_shape)))
        yy.append(torch.Tensor(crop_to_standard(gt, max_shape)))
    
    yy_hat = torch.stack(yy_hat, dim=0)
    yy = torch.stack(yy, dim=0)
    # print(max_shape)
    # print(yy_hat.shape, yy.shape)
    avg_dice_micro = tf.dice(yy_hat.long(), yy.long(),average='micro', ignore_index=0, num_classes=10)
    avg_dice_macro = tf.dice(yy_hat.long(), yy.long(),average='macro', ignore_index=0, num_classes=10)
    class_dice = tf.dice(yy_hat.long(), yy.long(), average=None, num_classes=10).numpy()
    metrics = {
        f'avg_dice_micro': avg_dice_micro.item(),
        f'avg_dice_macro': avg_dice_macro.item(),
        f'dice_lkidbey': class_dice[1],
        f'dice_rkidney': class_dice[2],
        f'dice_liver': class_dice[3],
        f'dice_spleen': class_dice[4],
        f'dice_llung': class_dice[5],
        f'dice_rlung': class_dice[6],
        f'dice_pancreas': class_dice[7],
        f'dice_gbladder': class_dice[8],
        f'dice_aorta': class_dice[9],
    }

    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from mostoolkit.ct_utils import crop_to_standard
    from mostoolkit.io_utils import sitk_load_with_metadata

    parser = ArgumentParser()
    parser.add_argument('-p', '--prediction', type=str, required=True)
    parser.add_argument('-gt', '--ground_truth', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, default='amos')           # either amos or dect
    args = parser.parse_args()

    pred = []
    gt = []
    max_shape = [0, 0, 0]
    for f in Path(args.prediction).glob('*.nii.gz'):
        assert (Path(args.ground_truth) / f.name).exists(), '{} has no ground truth'.format(f.name)
        ppred = sitk_load_with_metadata((str(f)))[0].astype(np.uint8)
        ggt = sitk_load_with_metadata(str(Path(args.ground_truth) / f.name))[0].astype(np.uint8)

        for i in range(3):
            if ppred.shape[i] > max_shape[i]:
                max_shape[i] = ppred.shape[i]

        pred.append(torch.Tensor(ppred))
        gt.append(torch.Tensor(ggt))
    print(max_shape)
    # pred = torch.cat(pred, dim=0)
    # gt = torch.cat(gt, dim=0)
    metrics = dict()
    if args.mode == 'ctorg':
        metrics = calculate_metrics_ctorg(pred, gt, max_shape)

    if args.mode == 'amos':
        metrics = calculate_metrics_amos(pred, gt, max_shape)
    
    if args.mode == 'dect':
        metrics = calculate_metrics_dect(pred, gt, max_shape)
    
    for k in metrics.keys():
        pp = format(metrics[k]*100, '.1f')
        print(k, ': ', pp)

# Note for cli: 
#  python ./metrics.py -p ./nnunet_raw/Dataset001_amos40/OutputsTs -gt ./nnunet_raw/Dataset001_amos40/labelsTs/ -m amos