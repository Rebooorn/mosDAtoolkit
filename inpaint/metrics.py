import torchmetrics.functional as tf
import torch
from mostoolkit.vis_utils import slice_visualize_XY
import numpy as np

def calculate_metrics(gt, out, mask):
    # mean dice and organ-wise dice: bg, bladder, bone, liver, lung, kidney,
    # 
    overall_rmse = tf.mean_squared_error(gt, out)
    hole_rmse = tf.mean_squared_error(gt[mask==0], out[mask==0])
    remainder_rmse = tf.mean_squared_error(gt[mask==1], out[mask==1])
    metrics = {
        'rmse': overall_rmse,
        'hole_rmse': hole_rmse,
        'remainder_rmse': remainder_rmse
    }
    return metrics

if __name__ == '__main__':
    import numpy as np
    from dataset import inpaint_dm
    
    dm = inpaint_dm(
        dataroot=r'D:\Chang\anaug\data128',
        masksroot=r'D:\Chang\anaug\inpaint\masks',
        batch_size=2
    )


    loader = dm.train_dataloader()

    for item in loader:
        gt = item['image']
        mask = item['mask']
        print(gt.shape)
        metrics = calculate_metrics(gt, torch.flip(gt, dims=[0,]), mask, stage='debug')
        print(metrics)
        break
        