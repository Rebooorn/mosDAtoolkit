import torch
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from net import PConvUNet as UNet
from metrics import calculate_metrics
from dataset import inpaint_dm
from inpaint_utils import parse_trainer_args
from mostoolkit.torch_utils import parse_loss_function_by_name
from mostoolkit.io_utils import load_yaml, save_yaml, sitk_save, combine_and_save_to_png
from skimage.transform import resize
import logging
from pathlib import Path
from mostoolkit.vis_utils import slice_visualize_X2Y2
import os, sys
import numpy as np
from loss import InpaintingLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class inpaint_model(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        # set up network
        config = self.hparams['model_config']
        self.ckpt_root = Path(self.hparams['ckptroot'])
        self.inpaint_network = UNet()

        # if self.hparams['pretrained'] is not None:
        #     # ckpt = torch.load(self.hparams['pretrained'])
        #     # print(ckpt['state_dict'].keys())
        #     # state_dict = {k.replace('inpaint_network.', ''): ckpt['state_dict'][k] for k in ckpt['state_dict'].keys()}

        #     ckpt = torch.load(r'./inpaint_repo/1065283/inpaint.pth')
        #     logging.info('Loading pretrained weights from: {}'.format(self.hparams['pretrained']))
        #     self.inpaint_network.load_state_dict(ckpt)

        # set up loss function
        self.loss = InpaintingLoss()
        self.val_step_outputs = []
        self.train_step_output = []
        self.test_step_output = []

        self.lambda_valid = 1.0
        self.lambda_hole = 6.0
        self.lambda_tv = 0.1
               
    def training_step(self, batch, batch_idx):
        gt, mask = batch['image'], batch['mask']
        input = gt * mask
        # print(x.shape)
        # print(input.shape, mask.shape)
        out, out_mask = self.inpaint_network(input, mask)
        loss_dict = self.loss(input, mask, out, gt)
        loss = loss_dict['valid'] * self.lambda_valid + loss_dict['hole'] * self.lambda_hole + loss_dict['tv'] * self.lambda_tv
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/loss_tv', loss_dict['tv'], prog_bar=False, sync_dist=True)
        self.log('train/loss_valid', loss_dict['valid'], prog_bar=False, sync_dist=True)
        self.log('train/loss_hole', loss_dict['hole'], prog_bar=False, sync_dist=True)

        self.train_step_output.append(loss.detach())
        if batch_idx == 0:
            save_fname = self.ckpt_root / 'train_visual' / 'epoch_{}.png'.format(self.current_epoch)
            self.save_pred_and_gt(gt, out, save_fname)
        return loss

    def validation_step(self, batch, batch_idx):
        gt, mask = batch['image'], batch['mask']
        # print(x.shape)
        input = gt * mask
        out, out_mask = self.inpaint_network(input, mask)
        # logging.info('min/max: {}/{}'.format(torch.min(out).item(), torch.max(out).item()))
        loss_dict = self.loss(input, mask, out, gt)
        loss = loss_dict['valid'] * self.lambda_valid + loss_dict['hole'] * self.lambda_hole + loss_dict['tv'] * self.lambda_tv
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        self.log('val/loss_tv', loss_dict['tv'], prog_bar=False, sync_dist=True)
        self.log('val/loss_valid', loss_dict['valid'], prog_bar=False, sync_dist=True)
        self.log('val/loss_hole', loss_dict['hole'], prog_bar=False, sync_dist=True)

        metrics = calculate_metrics(gt, out, mask)
        metrics['loss'] = loss.detach()
        self.val_step_outputs.append(metrics)

        if batch_idx == 0:
            save_fname = self.ckpt_root / 'valid_visual' / 'epoch_{}.png'.format(self.current_epoch)
            self.save_pred_and_gt(gt, out, save_fname)
        return metrics
    
    def save_pred_and_gt(self, gt, pred, fname):
        def scale_zero_one(var):
            return (var - var.min()) / (var.max() - var.min() + 1e-5)
        d = gt.shape[3]
        gt = resize(scale_zero_one(gt[0,0,:,d//2]).cpu().numpy(), (128, 128), order=0, preserve_range=True, anti_aliasing=False)
        pred = resize(scale_zero_one(pred[0,0,:,d//2]).cpu().detach().numpy(), (128, 128), order=0, preserve_range=True, anti_aliasing=False)
        vis = np.stack([gt[np.newaxis,:], pred[np.newaxis,:]], axis=0)
        combine_and_save_to_png(vis, fname)
    
    def save_prediction(self, pred, idx):
        save_path = Path(self.logger.log_dir) / 'test_model_predict' / f'{idx}.nii.gz'
        pred = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
        sitk_save(str(save_path), pred, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
      
    # def test_step(self, batch, batch_idx):
    #     x, y = batch['image'], batch['label']
    #     y_hat = self.seg_network(x)
    #     metrics = calculate_metrics(y_hat, y.long(), stage='test')
    #     self.test_step_output.append(metrics)
    #     return None
    
    def on_train_epoch_end(self) -> None:
        train_epoch_loss = torch.stack(self.train_step_output)
        train_epoch_loss = torch.mean(train_epoch_loss)
        self.log('train/epoch_loss', train_epoch_loss, sync_dist=True)
        self.train_step_output.clear()
        logging.info("Training\tepoch\t{}\tdone.".format(self.current_epoch))

    def on_validation_epoch_end(self) -> None:
        keys = self.val_step_outputs[0].keys()
        for k in keys:
            m = [i[k] for i in self.val_step_outputs]
            epoch_m = torch.mean(torch.stack(m))
            self.log(f'val/epoch_{k}', epoch_m.detach(), sync_dist=True)
        self.val_step_outputs.clear()
        logging.info("Validation\tepoch\t{}\tdone.".format(self.current_epoch))

    # def on_test_epoch_end(self) -> None:
    #     keys = self.test_step_output[0].keys()
    #     for k in keys:
    #         m = [i[k] for i in self.test_step_output]
    #         epoch_m = torch.mean(torch.stack(m))
    #         self.log(f'test/epoch_{k}', epoch_m.detach(), sync_dist=True)
    #         print(f'test/epoch_{k}: ', epoch_m.detach().item())
    #     self.test_step_output.clear()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.inpaint_network.parameters(), **self.hparams['model_config']['opt_args'])

def model_train():
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--masksroot', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--config', type=str, default=r'./inpaint/inpaint.yaml')

    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)
    config = load_yaml(dict_args['config'])
    config['data_config']['dataroot'] = dict_args['dataroot']
    config['data_config']['masksroot'] = dict_args['masksroot']
    
    logger = TensorBoardLogger(save_dir='tb_logs', name='inpaint_pconv', version=int(os.getenv('SLURM_JOBID') or 0))
    callbacks = [ModelCheckpoint(monitor='val/epoch_rmse', mode='min'),]
    trainer_args = parse_trainer_args(config['trainer'])
    trainer = pl.Trainer(**trainer_args, callbacks=callbacks, logger=logger)

    ckpt_root = Path(r'tb_logs') / 'inpaint_pconv' / 'version_{}'.format(trainer.logger.version)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    (ckpt_root / 'valid_visual').mkdir(exist_ok=True)
    (ckpt_root / 'train_visual').mkdir(exist_ok=True)
    config['ckptroot'] = str(ckpt_root)
    save_yaml(str(ckpt_root / 'config.yaml'), config)
    config['pretrained'] = args.pretrained if len(args.pretrained) > 0 else None

    model = inpaint_model(**config)
    datamodule = inpaint_dm(**config['data_config'])

    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule=datamodule)

def model_test():
    parser = ArgumentParser()
    parser.add_argument('--ckptpath', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    args = parser.parse_args()

    hparams = load_yaml(Path(args.ckptpath).parent.parent / 'hparams.yaml')
    data_config = hparams['data_config']
    data_config['dataroot'] = args.dataroot

    model = inpaint_model.load_from_checkpoint(
        checkpoint_path=args.ckptpath
    )

    dm = inpaint_dm(**data_config)

    trainer = pl.Trainer()
    trainer.test(model, dataloaders=dm.train_dataloader())

def model_save(v):
    import shutil
    ckptroot = r'./tb_logs/inpaint_pconv/version_{}/'.format(v)
    config = load_yaml(str(Path(ckptroot)/'config.yaml'))
    ckpt = list((Path(ckptroot)/'checkpoints').glob('*.ckpt'))[0]
    model = inpaint_model(**config).load_from_checkpoint(str(ckpt))
    save_path = r'./inpaint_repo/{}'.format(v)
    Path(save_path).mkdir(exist_ok=True)
    torch.save(model.inpaint_network.state_dict(), str(Path(save_path)/'inpaint.pth'))
    shutil.copy(str(Path(ckptroot)/'config.yaml'), save_path+r'/config.yaml')
    print('done')

    config = load_yaml(r'./inpaint_repo/{}/config.yaml'.format(v))
    net = UNet().load_state_dict(torch.load(r'./inpaint_repo/{}/inpaint.pth'.format(v)))

if __name__ == '__main__':
    model_train()
    # model_test()

    # model_save(1082574)