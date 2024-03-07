import numpy as np
from pathlib import Path
from mostoolkit.io_utils import sitk_load_with_metadata, sitk_save, save_json
from mostoolkit.ct_utils import crop_to_standard
from utils import roi_bbox, calculate_offset_fast, shift, volume_to_label_fname, print_n_estimation
# from tqdm import tqdm
import multiprocessing
from contextlib import closing
import logging
import sys
import random
from copy import deepcopy
from scipy import ndimage

# debug
# from mostoolkit.vis_utils import slice_visualize_X

'''
Difference from V2:
now the shifted mask will be dilated before transplanting the input organ to original organ.
'''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def augment_plan(dataroot, nclass, train_list, num_augment, tolerance=0.02, organ_list=None, multimodality=False):
    ''' nclass: include background '''
    root = Path(dataroot)

    # this is for multi-modality, 
    labels = [root/volume_to_label_fname(i, multimodality) for i in train_list]
    
    organ_size = dict()
    for l in labels:
        mask = sitk_load_with_metadata(str(l))[0]
        organ_size[l.name] = [np.sum(mask==i) for i in range(nclass)]
    # print(organ_size)
    filtered = dict()
    if organ_list is None:
        organ_list = list(range(1,nclass))
    for organ in organ_list:
        filtered[organ] = dict()
        for n, l in enumerate(labels):
            base_organsize = organ_size[l.name]
            if base_organsize[organ] == 0:
                # picked = [l.name,]
                picked = [n,]
            else:
                # tol = [max(i*tolerance, 100) for i in base_organsize]
                tol = [int(i*tolerance) for i in base_organsize]
                # when the organ is small, the tolerance will be non-sense
                # picked = [ll.name for ll in labels if organ_size[ll.name][organ] <= base_organsize[organ] + tol[organ] and organ_size[ll.name][organ] >= base_organsize[organ] - tol[organ]]
                picked = [nn for nn, ll in enumerate(labels) if organ_size[ll.name][organ] <= base_organsize[organ] + tol[organ] 
                                                            and organ_size[ll.name][organ] >= base_organsize[organ] - tol[organ]
                                                            and organ_size[ll.name][organ] > 0]
            # print('For {}: {} matched'.format(l.name, len(picked)))
            filtered[organ][n] = deepcopy(picked)
    f = []
    for n,l in enumerate(labels):
        ff = [len(filtered[o][n]) for o in organ_list]
        f.append(np.prod(ff))
    print(f)
    print('tolerance=', tolerance)
    print('Original num combination: ', len(organ_size.keys())**nclass)
    print('Filtered num combination: ', sum(f))

    aug_plan = []
    rd = random.Random(100)
    for _ in range(num_augment):
        base_ind = rd.randrange(0, len(labels))
        base = root/train_list[base_ind]
        # base_label = labels[base_ind]
        organs_ind = [rd.choice(filtered[i][base_ind]) for i in organ_list]
        # organs = [filtered[i][base_label][organ_ind[i]] for i in organ_ind]
        organs = [
            root/train_list[i] for i in organs_ind
        ]
        aug_plan.append({
            'base': str(base),
            'organs': [str(i) for i in organs]
        })
    return aug_plan



def inpaint_image(src, aug, src_label, aug_label, organ):
    src_mask = (src_label == organ).astype(np.float32)
    aug_mask = (aug_label == organ).astype(np.float32)

    # pad or crop aug_mask to the shape of src_mask
    # aug_mask = crop_to_standard(aug_mask, src_mask.shape)
    # aug = crop_to_standard(aug, src.shape)
    # Sometimes the organ region on the boundary will also be cropped, 
    # Now the organ roi will be cropped first
    if np.sum(src_mask) == 0 or np.sum(aug_mask) == 0:
        logging.info('missing organ in src, no augmentation can be done')
        return src, src_mask
    # print('>>', np.sum(src_mask), np.sum(aug_mask))
    roi = roi_bbox(aug_mask)

    roi_mask = np.zeros_like(aug_mask)
    roi_mask[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]] = 1.0

    aug_mask = aug_mask[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    aug = aug[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    roi_mask = roi_mask[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

    aug_mask = crop_to_standard(aug_mask, src_mask.shape)
    aug = crop_to_standard(aug, src.shape)
    roi_mask = crop_to_standard(roi_mask, src.shape)
    # print(np.sum(src_mask), np.sum(aug_mask))

    if np.sum(src_mask) == 0 or np.sum(aug_mask) == 0:
        logging.info('missing organ in src or aug after crop_to_standard')
        return src, src_mask

    dx, dy, dz = calculate_offset_fast(src_mask, aug_mask, pbar=False)

    struct = ndimage.generate_binary_structure(3, 1)
    # remove the src organ
    # masked_aug = aug * aug_mask
    # masked_aug = aug * (ndimage.binary_dilation(aug_mask, structure=struct).astype(np.float32) * roi_mask)
    shifted_aug = shift(aug,  (dx, dy, dz), order=0, prefilter=False)
    roi_mask = shift(roi_mask, (dx, dy, dz), order=0, prefilter=False)
    # shifted_aug = shift(aug,  (dx, dy, dz), order=0, prefilter=False)
    shifted_aug_mask = shift(aug_mask,  (dx, dy, dz), order=0, prefilter=False)
    # inpaint_mask = shifted_aug_mask * src_mask
    inpaint_mask = np.logical_or(shifted_aug_mask, src_mask)
    inpaint_src = deepcopy(src)
    # inpaint_src[src_mask>0] = 0

    # dilate the shifted_aug_mask a bit to keep the boundary
    transplant_mask = shifted_aug_mask > 0
    transplant_mask = ndimage.binary_dilation(transplant_mask, structure=struct, iterations=2).astype(np.float32)
    transplant_mask *= roi_mask

    inpaint_src[transplant_mask>0] = shifted_aug[transplant_mask>0]

    # inpaint_src = src + shifted_aug * inpaint_mask
    return inpaint_src, inpaint_mask


def anatomy_augment(data_root, src_list, save_path, nclass, num_augment, num_workers=0, organ_list=None, multimodality=False):

    # initial estimation:
    n_src = len(src_list)
    print_n_estimation(n_src, nclass)
    
    # We will not augment all the combinations, so specific amounts will be sampled
    # Note filter: not all combinations create in-scope volumes: we will keep only:
    #               1. The size diff < 20%
    aug_plan = augment_plan(data_root, nclass, src_list, num_augment, organ_list=organ_list, multimodality=multimodality)
    save_json(r'./anatomix_v2_plan.json', aug_plan)
    # return None
    

    # augment according to aug plan
    # first we create some deformed cached
    _, d, s, o, _ = sitk_load_with_metadata(str(aug_plan[0]['base']))

    if num_workers == 0:
        for n, plan in enumerate(aug_plan):
            print(n, plan)
            anatomy_augment_worker_fn(plan, n, nclass, d, s, o, save_path, organ_list, multimodality)
    else:
        args = [(plan, n, nclass, d, s, o, save_path, organ_list, multimodality) for n, plan in enumerate(aug_plan) ]
        with closing(multiprocessing.Pool(processes=num_workers)) as p:
            res = p.starmap(anatomy_augment_worker_fn, args)

def anatomy_augment_worker_fn(plan, n, n_class, direction, spacing, origin, save_path, organ_list, multimodality):
    d = direction
    s = spacing
    o = origin
    image_basename = Path(plan['base']).name
    label_basename = volume_to_label_fname(image_basename, multimodality)
    # print(label_basename)

    if (Path(save_path) / image_basename.replace('.nii.gz', '_{}.nii.gz'.format(n))).exists():
        logging.info('{} done'.format(image_basename.replace('.nii.gz', '_{}.nii.gz'.format(n))))
        return

    image_fname = Path(plan['base']).parent / image_basename
    label_fname = Path(plan['base']).parent / label_basename
    image_base = deepcopy(sitk_load_with_metadata(image_fname)[0])
    label_base = deepcopy(sitk_load_with_metadata(str(label_fname))[0])

    # image_base
    if organ_list is None:
        organ_list = range(n_class - 1)
    else:
        organ_list = [i-1 for i in organ_list]

    for nn, i in enumerate(organ_list):
        image_fname = plan['organs'][nn]
        label_fname = volume_to_label_fname(image_fname, multimodality)
        image_aug = deepcopy(sitk_load_with_metadata(image_fname)[0])
        label_aug = deepcopy(sitk_load_with_metadata(label_fname)[0])
        # print(image_aug.shape, label_aug.shape)
        image_base, inpaint_mask = inpaint_image(src=image_base, 
                                    aug=image_aug, 
                                    src_label=label_base, 
                                    aug_label=label_aug, 
                                    organ=i+1)
        label_base[label_base==i+1] = 0
        label_base[inpaint_mask==1] = i+1
        # if n == 3:
        #     sitk_save(r'./debug/{}-{}-organ{}.nii.gz'.format(image_basename, n, i), image_base, s, o, d)
        # if n == 3 and i == 4:
        #     print('checkpoint!')
        
    sitk_save(str(Path(save_path) / image_basename.replace('.nii.gz', '_{}.nii.gz'.format(n))), image_base, s, o, d)
    sitk_save(str(Path(save_path) / label_basename.replace('.nii.gz', '_{}.nii.gz'.format(n))), label_base, s, o, d)
    logging.info('{} done'.format(image_basename.replace('.nii.gz', '_{}.nii.gz'.format(n))))

def extract_holes(orig_mask, aug_mask):
    # return a mask where the holes are 1
    ori_mask_binary = orig_mask > 0
    aug_mask_bg = aug_mask == 0
    hole_mask = np.logical_and(ori_mask_binary, aug_mask_bg)
    return hole_mask
    
if __name__ == '__main__':

    import shutil
    from mostoolkit.io_utils import load_json
    from argparse import ArgumentParser
    parser = ArgumentParser(description='CAUTION: the images should be resized to the same pixel size')
    parser.add_argument('-sp', '--split', type=str, default='')             # the path to data split json.
    parser.add_argument('-p', '--plan', type=str, default='')
    parser.add_argument('-d','--data_root', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    parser.add_argument('-n', '--num_augment', type=int, default=500)
    parser.add_argument('-nc', '--nclass', type=int, required=True)         # does not include background.
    parser.add_argument('-nw', '--num_worker', type=int, default=0)         # num workers to accelerate the augmentation. 0 will not use multi-processes       
    parser.add_argument('-ipt', '--inpaint_version', type=int, default=1082574)
    parser.add_argument('--debug', action='store_true', default=False)      # if true, the intermediate results will be kept.
    parser.add_argument('-mm', '--multimodality', action='store_true', default=False)      # if true, the .
    parser.add_argument('-ro', '--reduced_organs', action='store_true', default=False)      # if true, the .
    parser.add_argument('-c', '--clean', action='store_true', default=False)      # if true, the clean previous data
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

    save_path = args.save_path

    if args.clean:
        shutil.rmtree(save_path)

    Path(save_path).mkdir(exist_ok=True)
    if args.reduced_organs:
        organ_list = [1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15]
    else:
        organ_list = None
    # tmp_dir = Path(args.save_path).parent / (Path(args.save_path).name+'_tmp')
    # tmp_dir.mkdir(exist_ok=True)
    # print(args.multimodality)
    anatomy_augment(data_root=args.data_root,
                    src_list=train_list, 
                    save_path=str(save_path), 
                    nclass=args.nclass,
                    num_augment=args.num_augment, 
                    organ_list=organ_list,
                    num_workers=args.num_worker,
                    multimodality=args.multimodality)

    # inpaint_holes(inpaint_version=args.inpaint_version, 
    #               root=str(tmp_dir), 
    #               save_path=args.save_path,
    #               orig_root=args.data_root,
    #               multimodality=args.multimodality)
    
    # if not args.debug:
        # shutil.rmtree(str(tmp_dir))
