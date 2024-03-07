from mostoolkit.io_utils import load_json, save_json, sitk_load_with_metadata, sitk_save
import numpy as np
import random
from mostoolkit.ct_utils import crop_to_standard
from argparse import ArgumentParser
from utils import *

def augment_plan(train_list, root, beta, num_augment=500):
    def random_roi(shape, lam):
        D = shape[0]
        W = shape[1]
        H = shape[2]
        cut_rat = np.sqrt(1. - lam)
        cut_d = int(D * cut_rat)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cz = np.random.randint(D)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbz1 = int(np.clip(cz - cut_d // 2, 0, D))
        bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
        bby1 = int(np.clip(cy - cut_h // 2, 0, H))
        bbz2 = int(np.clip(cz + cut_d // 2, 0, D))
        bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
        bby2 = int(np.clip(cy + cut_h // 2, 0, H))

        return [bbz1, bbx1, bby1, bbz2, bbx2, bby2]
    
    plan = []
    rd = random.Random(123)
    fnames = train_list
    for _ in range(num_augment):
        # selected image will be cropped to the size of the base image
        base = rd.choice(fnames)
        fnames_ = [i for i in fnames if i is not base]
        aug = rd.choice(fnames_)
        lam = np.random.beta(beta, beta)

        base_shape = sitk_load_with_metadata(str(Path(root)/base))[0].shape
        plan.append({
            'base': base,
            'aug': aug,
            'roi': random_roi(base_shape, lam)
        })
    return plan

def cutmix_augment(train_list, root, beta, save_path, num_augment, num_worker, multimodality=False):

    plan = augment_plan(train_list, root, beta, num_augment)
    save_json(r'./cutmix_plan.json', plan)
    root = Path(root)
    Path(save_path).mkdir(exist_ok=True)
    total = len(plan)
    for n,p in enumerate(plan):
        roi = p['roi']
        base = str(root / p['base'])
        aug = str(root / p['aug'])
        base, d, s, o, _ = sitk_load_with_metadata(base)
        aug = sitk_load_with_metadata(aug)[0]

        # base_mask = str(root / p['base'].replace('volume', 'labels'))
        # aug_mask = str(root / p['aug'].replace('volume', 'labels'))
        base_mask = root/volume_to_label_fname(p['base'], multimodality)
        aug_mask = root/volume_to_label_fname(p['aug'], multimodality)
        base_mask = sitk_load_with_metadata(str(base_mask))[0]
        aug_mask = sitk_load_with_metadata(str(aug_mask))[0]

        aug = crop_to_standard(aug, base.shape)
        aug_mask = crop_to_standard(aug_mask, base.shape)
        mask = np.zeros(base.shape)
        mask[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]] = 1    
        res = base * (1-mask) + aug * mask
        res_mask = base_mask * (1-mask) + aug_mask * mask
        
        tar_imgfname = str(Path(save_path) / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)))
        tar_maskfname = volume_to_label_fname(tar_imgfname, multimodality)
        # tar_maskfname = str(save_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)).replace('volume', 'labels'))
        sitk_save(tar_imgfname, res, s, o, d)
        sitk_save(tar_maskfname, res_mask, s, o, d)
        print(f'{n}/{total} done')

if __name__ == '__main__':
    import shutil
    from pathlib import Path
    from mostoolkit.io_utils import load_json
    parser = ArgumentParser()
    parser.add_argument('-sp', '--split', type=str, default='')             # the path to data split json.
    parser.add_argument('-d', '--data_root', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    parser.add_argument('-n', '--num_augment', type=int, default=500)
    parser.add_argument('-b', '--beta', type=float, default=1.0)
    parser.add_argument('-nw', '--num_worker', type=int, default=0)         # num workers to accelerate the augmentation. 0 will not use multi-processes       
    parser.add_argument('-mm', '--multimodality', action='store_true', default=False)      # if true, the .
    args = parser.parse_args()

    split = args.split

    assert Path(args.data_root).exists(), 'data root cannot be found, input: {}'.format(args.data_root)

    if split == '':
        # no split.json is given
        print('>> No --split is given, all data in the --data_root will be used for augmentation')
        train_list = [i.name for i in Path(args.data_root).glob('volume*.nii.gz')]
    else:
        all_data = load_json(split)
        train_list = all_data['train']


    cutmix_augment(train_list, args.data_root, args.beta, args.save_path, args.num_augment, args.num_worker, args.multimodality)
