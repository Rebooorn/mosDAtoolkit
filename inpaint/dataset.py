import pytorch_lightning as pl
import monai.transforms as mtf
from pathlib import Path
from mostoolkit.io_utils import load_json, save_json
from monai.data.dataset import Dataset, CacheDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

def get_train_transforms(aug=False):
    keys = ['image', 'mask']
    transforms = [
        mtf.LoadImaged(keys, image_only=True),
        mtf.EnsureChannelFirstd(keys),
        mtf.SpatialPadd(keys[:1], (128,128,128)),
        mtf.RandSpatialCropd(keys[:1], roi_size=(128,128,128), random_size=False),
        mtf.ScaleIntensityRanged(keys[:1], a_min=0, a_max=2800, b_min=0, b_max=1, clip=True),
        # mtf.SpatialPadd(keys, spatial_size=[128,128,128])
        mtf.RandFlipd(keys=keys[:1], prob=0.3, spatial_axis=0),
        mtf.RandFlipd(keys=keys[:1], prob=0.3, spatial_axis=1),
        mtf.RandFlipd(keys=keys[:1], prob=0.3, spatial_axis=2)
    ]
    # if aug:
    #     transforms.extend([
    #         mtf.
    #     ])

    transforms.append(mtf.ToTensord(keys))
    return mtf.Compose(transforms)

def get_val_transforms():
    keys = ['image', 'mask']
    transforms = [
        mtf.LoadImaged(keys, image_only=True),
        mtf.EnsureChannelFirstd(keys),
        mtf.SpatialPadd(keys[:1], (128,128,128)),
        mtf.RandSpatialCropd(keys[:1], roi_size=(128,128,128), random_size=False),
        mtf.ScaleIntensityRanged(keys[:1], a_min=0, a_max=2800, b_min=0, b_max=1, clip=True),
        # mtf.SpatialCropd(keys, roi_center=[64, 40, 40], roi_size=[128, 128, 128]),
        mtf.ToTensord(keys),
    ]
    return mtf.Compose(transforms)

# def get_test_transforms():
#     keys = ['image', 'label']
#     transforms = [
#         mtf.LoadImaged(keys, image_only=True),
#         mtf.EnsureChannelFirstd(keys),
#         mtf.ScaleIntensityRanged(keys[:1], a_min=700, a_max=1300, b_min=0, b_max=1, clip=True),
#         mtf.SpatialCropd(keys, roi_center=[64, 40, 40], roi_size=[128, 128, 128]),
#         # mtf.SpatialCropd(keys, roi_size=[128, 128, 128]),
#         mtf.ToTensord(keys),
#     ]
#     return mtf.Compose(transforms)

def generate_data_split(dataroot, masksroot):
    import random
    import numpy as np
    dataroot = Path(dataroot)
    masksroot = Path(masksroot)

    data = list(dataroot.glob('volume-*.nii.gz'))
    masks = list(masksroot.glob('*.nii.gz'))
    print(len(data), len(masks))

    data_ids = np.arange(len(data))
    rd = random.Random(123)
    rd.shuffle(data_ids)

    train_data = data_ids[:80]
    valid_data = data_ids[80:90]
    test_data = data_ids[90:]

    train_data = [data[i] for i in train_data]
    valid_data = [data[i] for i in valid_data]
    test_data = [data[i] for i in test_data]
    # masks = [str(i) for i in masks]

    split = {
        'train': [],
        'val': [],
        'test': [],
    }

    for i in range(10000):
        d = {
            'image': rd.choice(train_data).name,
            'mask': rd.choice(masks).name
        }
        split['train'].append(d)

    for i in range(len(valid_data)):
        d = {
            'image': valid_data[i].name,
            'mask': rd.choice(masks).name
        }
        split['val'].append(d)

    for i in range(len(test_data)):
        d = {
            'image': test_data[i].name,
            'mask': rd.choice(masks).name,
        }
        split['test'].append(d)

    save_json('inpaint_split.json', split)
    print('split.json saved')


class inpaint_dm(pl.LightningDataModule):
    def __init__(self, dataroot: str, masksroot: str, batch_size: int):
        super().__init__()
        # dataroot/(volume-x.nii.gz+labels-x.nii.gz)
        # dataroot/(split.json)
        # masksroot/000001.nii.gz

        self.train_transforms = get_train_transforms()
        self.val_transforms = get_val_transforms()

        self.dataroot = Path(dataroot)
        self.masksroot = Path(masksroot)

        self.data_split = load_json(r'./inpaint/inpaint_split.json')
        self.train_items = self.load_items(self.data_split['train'])
        self.val_items = self.load_items(self.data_split['val'])
        # self.test_items = self.load_items(self.data_split['test'], stage='test')

        self.batch_size = batch_size

    def load_items(self, index):
        ret = []
        for d in index:
            ret.append({
                'image': str(self.dataroot / d['image']),
                'mask': str(self.masksroot / d['mask']),
            })
        return ret
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = Dataset(self.train_items, self.train_transforms)
        # dataset = CacheDataset(self.train_items, self.train_transforms)
        return DataLoader(dataset, batch_size=self.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = Dataset(self.val_items, self.val_transforms)
        return DataLoader(dataset, batch_size=self.batch_size)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = Dataset(self.test_items, self.test_transforms)
        return DataLoader(dataset, batch_size=1)
    
def debug_inpaint_dataset(dataroot, masksroot):
    from mostoolkit.vis_utils import slice_visualize_XY

    dm = inpaint_dm(
        dataroot, masksroot, 2
    )

    loader = dm.train_dataloader()

    for item in loader:
        gt = item['image'].cpu().numpy()
        mask = item['mask'].cpu().numpy()

        slice_visualize_XY(gt[0,0], mask[0,0])




if __name__ == '__main__':
    # generate_data_split(
    #     dataroot=r'D:\Chang\anaug\data128',
    #     masksroot=r'D:\Chang\anaug\inpaint\masks'
    # )

    dataroot=r'D:\Chang\anaug\data128'
    masksroot=r'D:\Chang\anaug\inpaint\masks'
    debug_inpaint_dataset(dataroot, masksroot)
