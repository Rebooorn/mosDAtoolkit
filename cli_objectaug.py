from mostoolkit.io_utils import load_json, save_json, sitk_load_with_metadata, sitk_save
import numpy as np
import random
from mostoolkit.ct_utils import crop_to_standard
from monai.transforms import Rotate, Zoom
from scipy.ndimage import shift
import torch
from utils import *
from contextlib import closing
from argparse import ArgumentParser
import multiprocessing

'''
According to the original paper:
"Besides, we also used some traditional data augmentation methods including 
random scale, random shift, random rotation,  and random horizontal flip to augment objects in ObjectAug. "

because random flip makes no sense in medical image, we use only 
random scaling 0.8-1.2
random shift: 10
random rotation: 15 

'''

def augment_plan(train_list, root, nclass, num_augment=500, multimodality=False):
    plan = []
    rd = random.Random(123)
    fnames = train_list
    root = Path(root)
    for _ in range(num_augment):
        # selected image will be cropped to the size of the base image
        base = rd.choice(fnames)
        basemask = volume_to_label_fname(root/base, multimodality)
        pplan = {
            'base': base,
        }
        rl = 15 * np.pi / 180
        shft = 3
        for i in range(nclass):
            pplan[f'obj{i}'] = {
                'scale': rd.random()*0.2 + 0.9,
                'shift': [rd.randint(-shft,shft), rd.randint(-shft,shft), rd.randint(-shft,shft)],
                'rotate': [rd.random()*rl*2.0-rl,rd.random()*rl*2.0-rl,rd.random()*rl*2.0-rl]
            }
        plan.append(pplan)
    return plan


class inpainter:
    def __init__(self, model_version=1082574) -> None:
        self.model = load_inpaint_pconv_model(model_version).cuda()

    def extract_holes(self, orig_mask, aug_mask):
        # return a mask where the holes are 1
        ori_mask_binary = orig_mask > 0
        aug_mask_bg = aug_mask == 0
        hole_mask = np.logical_and(ori_mask_binary, aug_mask_bg)
        return hole_mask

    def forward(self, img, orig_mask, aug_mask):
        img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
        img = img / 2800.0          # already +1024
        # orig_shape = img.shape
        hole_mask = 1 - self.extract_holes(orig_mask, aug_mask)
        

        inpainted, _ = self.model(img.cuda(), torch.Tensor(hole_mask).unsqueeze(0).unsqueeze(0).cuda())
        masked_inpainted = deepcopy(img[0,0].cpu().numpy())
        inpainted = inpainted[0,0].detach().cpu().numpy()
        # print(masked_inpainted.shape)
        # print(hole_mask.shape)
        masked_inpainted[hole_mask==0] = inpainted[hole_mask==0]

        # masked_inpainted = crop_to_standard(masked_inpainted, orig_shape)
        masked_inpainted = (masked_inpainted * 2800).astype(np.int32)

        return masked_inpainted, aug_mask

def objectaug_worder_fn(n,p,root, tmp_dir, nclass, multimodality):
    base = str(root / p['base'])
    base, d, s, o, _ = sitk_load_with_metadata(base)
    base_mask = str(volume_to_label_fname(root/p['base'], multimodality))
    base_mask = sitk_load_with_metadata(base_mask)[0]

    res_image = base * (base_mask==0)
    res_mask = np.zeros_like(res_image).astype(np.uint8)

    for i in range(nclass):
        ind = i+1
        pplan = p[f'obj{i}']
        obj_mask = base_mask==ind
        masked_base = base * obj_mask

        rotate = Rotate(pplan['rotate'], dtype=None)
        zoom = Zoom(pplan['scale'], dtype=None)
        # print(masked_base.shape)
        masked_base = np.expand_dims(masked_base, axis=0)
        aug_image = rotate(masked_base)
        aug_image = zoom(aug_image)
        aug_image = aug_image[0]
        aug_image = shift(aug_image, pplan['shift'])
        
        # print(pplan['rotate'])
        rotate = Rotate(pplan['rotate'], mode='nearest', dtype=None)
        zoom = Zoom(pplan['scale'], mode='nearest', dtype=None)
        obj_mask = np.expand_dims(obj_mask, axis=0).astype(np.float32)
        aug_mask = rotate(obj_mask)
        aug_mask = zoom(aug_mask)
        aug_mask = aug_mask[0]
        aug_mask = shift(aug_mask, pplan['shift'])

        # now we move it to the background
        res_image[aug_mask==1] = aug_image[aug_mask==1]
        res_mask[aug_mask==1] = ind

    # save to tmp
    tar_imgfname = str(tmp_dir / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)))
    tar_maskfname = str(volume_to_label_fname(tar_imgfname, multimodality))
    sitk_save(str(tar_imgfname), res_image, s, o, d)
    sitk_save(str(tar_maskfname), res_mask, s, o, d)
    print('init {} done.'.format(Path(tar_imgfname).name))
    return 0
    

def objectaug_augment(train_list, root, nclass, save_path, num_augment, num_workers=0, inpaint_version=1082574, multimodality=False, plan=None):
    '''nclass should not include background'''
    if plan is None:
        plan = augment_plan(train_list, root, nclass, num_augment, multimodality=multimodality)
        save_json(r'./objectaug_plan.json', plan)
    else:
        plan = load_json(plan)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    tmp_path = save_path.parent / (save_path.name+'_tmp')
    tmp_path.mkdir(exist_ok=True)

    inpaint_worker = inpainter(model_version=inpaint_version)
    root = Path(root)
    total = len(plan)-1

    # if num_workers > 0:
    #     print('{} workers start working!'.format(num_workers))
    #     args = [(n,p,root, tmp_path, nclass, multimodality) for n, p in enumerate(plan)]
    #     with closing(multiprocessing.Pool(processes=num_workers)) as p:
    #         res = p.starmap(objectaug_worder_fn, args)

    # else:
    #     for n, p in enumerate(plan):
    #         objectaug_worder_fn(n,p, root, tmp_path, nclass, multimodality)

    for n,p in enumerate(plan):
        # base = str(root / p['base'])
        # base, d, s, o, _ = sitk_load_with_metadata(base)
        # base_mask = str(volume_to_label_fname(root/p['base'], multimodality))
        # base_mask = sitk_load_with_metadata(base_mask)[0]

        # res_image = base * (base_mask==0)
        # res_mask = np.zeros_like(res_image).astype(np.uint8)

        # for i in range(nclass):
        #     ind = i+1
        #     pplan = p[f'obj{i}']
        #     obj_mask = base_mask==ind
        #     masked_base = base * obj_mask

        #     rotate = Rotate(pplan['rotate'], dtype=None)
        #     zoom = Zoom(pplan['scale'], dtype=None)
        #     # print(masked_base.shape)
        #     masked_base = np.expand_dims(masked_base, axis=0)
        #     aug_image = rotate(masked_base)
        #     aug_image = zoom(aug_image)
        #     aug_image = aug_image[0]
        #     aug_image = shift(aug_image, pplan['shift'])
            
        #     # print(pplan['rotate'])
        #     rotate = Rotate(pplan['rotate'], mode='nearest', dtype=None)
        #     zoom = Zoom(pplan['scale'], mode='nearest', dtype=None)
        #     obj_mask = np.expand_dims(obj_mask, axis=0).astype(np.float32)
        #     aug_mask = rotate(obj_mask)
        #     aug_mask = zoom(aug_mask)
        #     aug_mask = aug_mask[0]
        #     aug_mask = shift(aug_mask, pplan['shift'])

        #     # now we move it to the background
        #     res_image[aug_mask==1] = aug_image[aug_mask==1]
        #     res_mask[aug_mask==1] = ind

        # inpaint the bg
        res_image = str(tmp_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)))
        res_mask = volume_to_label_fname(res_image, multimodality)
        res_image, d, s, o, _ = sitk_load_with_metadata(res_image)
        base_mask = volume_to_label_fname(root/p['base'], multimodality)
        base_mask = sitk_load_with_metadata(base_mask)[0]
        res_mask = sitk_load_with_metadata(res_mask)[0]
        res_image, _ = inpaint_worker.forward(res_image, base_mask, res_mask)

        tar_imgfname = str(save_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)))
        tar_maskfname = str(volume_to_label_fname(tar_imgfname, multimodality))
        # tar_maskfname = str(save_path / p['base'].replace('.nii.gz', '_{}.nii.gz'.format(n)).replace('volume', 'labels'))
        sitk_save(tar_imgfname, res_image, s, o, d)
        sitk_save(tar_maskfname, res_mask, s, o, d)
        print(f'{n}/{total} done')

if __name__ == '__main__':
    import shutil
    from mostoolkit.io_utils import load_json
    parser = ArgumentParser()
    parser.add_argument('-sp', '--split', type=str, default='')             # the path to data split json.
    parser.add_argument('-p', '--plan', type=str, default='')             # the path to plan json.
    parser.add_argument('-d', '--data_root', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    parser.add_argument('-n', '--num_augment', type=int, default=500)
    parser.add_argument('-nc', '--nclass', type=int, required=True)         # no background
    parser.add_argument('-ipt', '--inpaint_version', type=int, default=1082574)
    parser.add_argument('-nw', '--num_worker', type=int, default=0)         # num workers to accelerate the augmentation. 0 will not use multi-processes       
    parser.add_argument('--debug', action='store_true', default=False)      # if true, the intermediate results will be kept.
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

    if len(args.plan) == 0:
        # no plan input
        plan = None
    else:
        plan = args.plan

    objectaug_augment(train_list, 
                      args.data_root, 
                      args.nclass, 
                      args.save_path, 
                      args.num_augment,
                      num_workers=args.num_worker,
                      inpaint_version=args.inpaint_version, 
                      multimodality=args.multimodality,
                      plan=plan)
