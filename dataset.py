import pytorch_lightning as pl
import monai.transforms as mtf
from pathlib import Path
from mostoolkit.io_utils import load_json
from monai.data.dataset import Dataset, CacheDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from copy import deepcopy
import random

'''
Might be useful when you need some fast results...
'''

def get_train_transforms(aug=False):
    keys = ['image', 'label']
    transforms = [
        mtf.LoadImaged(keys, image_only=True),
        mtf.EnsureChannelFirstd(keys),
        mtf.ScaleIntensityRanged(keys[:1], a_min=0, a_max=2800, b_min=0, b_max=1, clip=True)
        # mtf.SpatialPadd(keys, spatial_size=[128,128,128])
    ]

    transforms.append(mtf.ToTensord(keys))
    return mtf.Compose(transforms)

def get_val_transforms():
    keys = ['image', 'label']
    transforms = [
        mtf.LoadImaged(keys, image_only=True),
        mtf.EnsureChannelFirstd(keys),
        mtf.ScaleIntensityRanged(keys[:1], a_min=0, a_max=2800, b_min=0, b_max=1, clip=True),
        # mtf.SpatialCropd(keys, roi_center=[64, 40, 40], roi_size=[128, 128, 128]),
        mtf.ToTensord(keys),
    ]
    return mtf.Compose(transforms)

def get_test_transforms():
    keys = ['image', 'label']
    transforms = [
        mtf.LoadImaged(keys, image_only=True),
        mtf.EnsureChannelFirstd(keys),
        mtf.ScaleIntensityRanged(keys[:1], a_min=0, a_max=2800, b_min=0, b_max=1, clip=True),
        # mtf.SpatialCropd(keys, roi_center=[64, 40, 40], roi_size=[128, 128, 128]),
        # mtf.SpatialCropd(keys, roi_size=[128, 128, 128]),
        mtf.ToTensord(keys),
    ]
    return mtf.Compose(transforms)



class basemodule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        # dataroot/(volume-x.nii.gz+labels-x.nii.gz)
        # dataroot/(split.json)

        self.train_transforms = get_train_transforms()
        self.test_transforms = get_test_transforms()
        self.val_transforms = get_val_transforms()

        self.batch_size = batch_size
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # dataset = Dataset(self.train_items, self.train_transforms)
        dataset = Dataset(self.train_items, self.train_transforms)
        return DataLoader(dataset, batch_size=self.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = Dataset(self.val_items, self.val_transforms)
        return DataLoader(dataset, batch_size=1)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = Dataset(self.test_items, self.test_transforms)
        return DataLoader(dataset, batch_size=1)
    
class anatomixmodule(basemodule):
    def __init__(self,batch_size, split, dataset_size=1000) -> None:
        super().__init__(batch_size)
        # train data will be globbed from the train_dataroot
        rd = random.Random(1000)
        print('Loaded data split: ', split)
        root = Path(r'./data128/full')
        train_list = load_json(split)['train']
        val_list = load_json(split)['val']
        val_list = [root/i for i in val_list]
        
        train_dataroot = r'./anatomix_final{}'.format(len(train_list))
        self.train_dataroot = Path(train_dataroot)
        train_items = list(self.train_dataroot.glob('volume*.nii.gz'))
        self.train_items = [{
            'image': str(k),
            'label': str(k).replace('volume', 'labels')
        } for k in train_items]

        # val_items = list(Path(r'./data128/valid').glob('volume*.nii.gz'))
        self.val_items = [{
            'image': str(k),
            'label': str(k).replace('volume', 'labels')
        } for k in val_list]
        
        self.test_items = deepcopy(self.val_items)
        print('Length of train/val dataset: {}/{}'.format(len(self.train_items), len(self.val_items)))

class normalmodule(basemodule):
    def __init__(self, batch_size, split,  dataset_size=1000):
        super().__init__(batch_size)
        rd = random.Random(1000)
        print('Loaded data split: ', split)
        root = Path(r'./data128/full')
        all_data = load_json(split)
        train_list = all_data['train']
        val_list = all_data['val']
        # train_list = [root/i for i in train_list]
        val_list = [root/i for i in val_list]

        self.train_items = []
        for _ in range(dataset_size):
            p = rd.choice(train_list)
            self.train_items.append({
                'image': str(root/p),
                'label': str(root/p).replace('volume', 'labels')
            })

        self.val_items = [{
            'image': str(k),
            'label': str(k).replace('volume', 'labels')
        } for k in val_list]
        
        self.test_items = deepcopy(self.val_items)
        print('Length of train/val dataset: {}/{}'.format(len(self.train_items), len(self.val_items)))

if __name__ == '__main__':
    print('debug dataloader')
    dataroot = '.'
    dm = anatomixmodule(train_dataroot=r'./anatomix_final', batch_size=1)
    # dm = normalmodule(batch_size=1)

    from mostoolkit.vis_utils import slice_visualize_XY
    import torch

    # loader = dm.train_dataloader()
    # for i in loader:
    #     print(i['image'].shape, torch.min(i['image']), torch.max(i['image']), torch.unique(i['label']))
    #     # slice_visualize_XY(i['image'][0], i['label'][0])

    # loader = dm.val_dataloader()
    # for i in loader:
    #     print(i['image'].shape, torch.min(i['image']), torch.max(i['image']), torch.unique(i['label']))
    #     # slice_visualize_XY(i['image'][0], i['label'][0])

    loader = dm.val_dataloader()
    for i in loader:
        print(i['image'].shape, torch.min(i['image']), torch.max(i['image']), torch.unique(i['label']))
        img = i['image'][0,0].cpu().numpy()
        seg = i['label'][0,0].cpu().numpy()
        slice_visualize_XY(img, seg)
