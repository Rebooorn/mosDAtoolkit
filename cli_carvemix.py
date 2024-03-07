from mostoolkit.io_utils import load_json, save_json, sitk_load_with_metadata, sitk_save
import numpy as np
import random
from mostoolkit.ct_utils import crop_to_standard
from copy import deepcopy
from scipy import ndimage
from argparse import ArgumentParser
from utils import volume_to_label_fname
import multiprocessing
from contextlib import closing

def get_distance(f,spacing):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -(dist_func(f,sampling=spacing)),
                        dist_func(1-f,sampling=spacing))

    return distance


def augment_plan(train_list, nclass, num_augment=500):
    '''nclass should be without background'''
    plan = []
    rd = random.Random(123)
    fnames = train_list
    for _ in range(num_augment):
        # selected image will be cropped to the size of the base image
        base = rd.choice(fnames)
        pplan = {'base': base,}
        for i in range(nclass):
            pplan['obj{}'.format(i)] = rd.choice(fnames)
        plan.append(pplan)
    return plan

def carvemix_augment_worker_fn(n, p, root, nclass, save_path, total, multimodality):
    base = str(root / p['base'])

    if (save_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n))).exists():
        print(f'{n}/{total} done')
        return 1
    base, d, s, o, _ = sitk_load_with_metadata(base)
    # base_mask = str(root / p['base'].replace('volume', 'labels'))
    base_mask = str(volume_to_label_fname(root/p['base'], multimodality))
    base_mask = sitk_load_with_metadata(base_mask)[0]

    res_img = deepcopy(base)
    res_mask = deepcopy(base_mask)

    for i in range(nclass):
        aug = p['obj{}'.format(i)]
        # aug_mask = str(root / aug.replace('volume', 'labels'))
        aug_img = str(root / aug)
        aug_mask = str(volume_to_label_fname(root/aug, multimodality))
        ind = i + 1
        aug_img = sitk_load_with_metadata(aug_img)[0]
        aug_mask = sitk_load_with_metadata(aug_mask)[0]

        aug_img = crop_to_standard(aug_img, base.shape)
        aug_mask = crop_to_standard(aug_mask, base_mask.shape)

        obj_base_mask = res_mask==ind
        obj_aug_mask = aug_mask == ind
        dist_aug = get_distance(obj_aug_mask, s)

        c = np.random.beta(1, 1)   # [0,1] creat distance
        c = (c-0.5)*2  # [-1.1] 
        if c>0:
            lam=c*np.min(dist_aug)/2              # λl = -1/2|min(dist_aug)|
        else:
            lam=c*np.min(dist_aug) 
                
        mask = (dist_aug<lam).astype('float32')   #creat M   

        res_img = res_img * (1-mask) + aug_img * mask
        obj_res_mask = obj_base_mask * (1-mask) + obj_aug_mask * mask
        res_mask[res_mask==ind] = 0
        res_img[res_mask==ind] = 0
        res_mask[obj_res_mask==1] = ind
        
    tar_imgfname = str(save_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)))
    # tar_maskfname = str(save_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)).replace('volume', 'labels'))
    tar_maskfname = str(volume_to_label_fname(tar_imgfname, multimodality))
    sitk_save(tar_imgfname, res_img, s, o, d)
    sitk_save(tar_maskfname, res_mask, s, o, d)
    print(f'{n}/{total} done')

def carvemix_augment(train_list, root, nclass, save_path, num_augment, num_workers, multimodality):

    plan = augment_plan(train_list, nclass, num_augment)
    save_json(r'./carvemix_plan.json', plan)
    root = Path(root)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    total = len(plan)-1
    if num_workers == 0:
        for n,p in enumerate(plan):
            carvemix_augment_worker_fn(n, p, root, nclass, save_path, total, multimodality)
    else:
        args = [(n, p, root, nclass, save_path, total, multimodality) for n, p in enumerate(plan) ]
        with closing(multiprocessing.Pool(processes=num_workers)) as p:
            res = p.starmap(carvemix_augment_worker_fn, args)
        

if __name__ == '__main__':
    from pathlib import Path
    from mostoolkit.io_utils import load_json
    parser = ArgumentParser()
    parser.add_argument('-sp', '--split', type=str, default='')             # the path to data split json.
    parser.add_argument('-d', '--data_root', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    parser.add_argument('-n', '--num_augment', type=int, default=500)
    parser.add_argument('-nc', '--nclass', type=int, required=True)         # no background
    parser.add_argument('-nw', '--num_worker', type=int, default=0)         # num workers to accelerate the augmentation. 0 will not use multi-processes       
    parser.add_argument('-mm', '--multimodality', action='store_true', default=False)      # if true, the .
    args = parser.parse_args()

    split = args.split
    assert Path(args.data_root).exists(), 'data root cannot be found, input: {}'.format(args.data_root)

    if split == '':
        # no split.json is given
        print('>> No --split is given, all data in the --data_root will be used for augmentation')
        train_list = list(Path(args.data_root).glob('volume*.nii.gz'))
    else:
        all_data = load_json(split)
        train_list = all_data['train']

    carvemix_augment(train_list, args.data_root, args.nclass, args.save_path, args.num_augment, args.num_worker, args.multimodality)
