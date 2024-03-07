import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.transform import rescale
from scipy.ndimage import shift, distance_transform_edt
from copy import deepcopy
import logging
from mostoolkit.io_utils import load_yaml
from monai.networks.nets.unet import UNet
import torch
from mostoolkit.ct_utils import crop_to_standard
from argparse import ArgumentParser

def volume_to_label_fname(fname, multimodality=False):
    # for multi-modality, the volume and label fname are volume_XXX_1.nii.gz and labels_XXX.nii.gz
    label = Path(fname).name
    comp = Path(fname).name.split('.')[0].split('_')
    comp[0] = 'labels'       # volume or labels
    # print(comp)
    if multimodality:
        comp.pop(2)             # modality
    label = '_'.join(comp) + '.nii.gz'
    return str(Path(fname).parent / label)

def parse_trainer_args(args: dict):
    # os.name == 'nt' ==> Windows ==> debug
    if os.name == 'nt':
        args['strategy'] = 'auto'
    else:
        args['strategy'] = 'ddp'
        args['enable_progress_bar'] = False
    return args

def print_n_estimation(n_img, n_class):
    n = (n_img) ** n_class
    print('>> Initial n estimation for anatomy augment:')
    print('>> {} images, {} classes(no bg)'.format(n_img, n_class))
    print('>> Augmented dataset size: {}'.format(n))

def mask_to_forground_index(mask):
    # convert the mask to the index of the foreground voxels
    ndim = len(mask.shape)
    if ndim==3:
        xx, yy, zz = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), np.arange(mask.shape[2]), indexing='ij')
        return xx, yy, zz
    elif ndim == 2:
        xx, yy = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
        return xx, yy
    else:
        return None

def calculate_mapping(src_mask, aug_mask):
    # When the src organ cannot be fully covered by the aug organ, the "holes" will be extended beyond the boundary using mirroring
    edt_aug, edt_index = distance_transform_edt(1-aug_mask, return_indices=True)
    holes_mask = src_mask * ((src_mask + aug_mask) == 1)
    xx, yy, zz = mask_to_forground_index(holes_mask)
    holes_ind = np.stack([xx[holes_mask], yy[holes_mask], zz[holes_mask]])

    mask_edt_aug = src_mask * edt_aug
    mask_edt_index = np.stack([src_mask,]*3, axis=0) * edt_index
    # mirror
    mirror_mapping = 2 * mask_edt_index - holes_ind

    return mirror_mapping, holes_mask

def calculate_offset_fast(src, aug, pbar=False):
    # a faster implementation of calculate_offset() using multi-resolution
    search_range = None
    for r in [0.125, 0.25, 0.5, 1.0]:
        src_lowres = rescale(src, r, order=0, preserve_range=True, anti_aliasing=False)
        aug_lowres = rescale(aug, r, order=0, preserve_range=True, anti_aliasing=False)
        if np.sum(src_lowres) == 0.0 or np.sum(aug_lowres) == 0.0:
            # downsampling can erase the small organs
            continue
        dx, dy, dz = calculate_offset(src_lowres, aug_lowres, search_range=search_range, pbar=pbar)
        search_range = [range(2*dx-2, 2*dx +2), range(2*dy-2, 2*dy +2), range(2*dz-2, 2*dz +2)]
    return dx, dy, dz

def calculate_offset(src, aug, search_range: list=None, pbar=False):
    xx, yy, zz = np.meshgrid(np.arange(src.shape[0]), np.arange(src.shape[1]), np.arange(src.shape[2]), indexing='ij')
    src_ = np.stack([xx[src==1], yy[src==1], zz[src==1]])
    xx_, yy_, zz_ = np.meshgrid(np.arange(aug.shape[0]), np.arange(aug.shape[1]), np.arange(aug.shape[2]), indexing='ij')
    aug_ = np.stack([xx_[aug==1], yy_[aug==1], zz_[aug==1]])

    if search_range is None:
        xrange = range(
            min(-np.min(aug_[0]), np.min(src_[0])-np.max(aug_[0])), 
            min(src.shape[0]-np.max(aug_[0]), np.max(src_[0])-np.min(aug_[0]))
            )
        yrange = range(
            min(-np.min(aug_[1]), np.min(src_[1])-np.max(aug_[1])), 
            min(src.shape[1]-np.max(aug_[1]), np.max(src_[1])-np.min(aug_[1]))
            )
        zrange = range(
            min(-np.min(aug_[2]), np.min(src_[2])-np.max(aug_[2])), 
            min(src.shape[2]-np.max(aug_[2]), np.max(src_[2])-np.min(aug_[2]))
            )
    else:
        xrange, yrange, zrange = search_range

    max_overlap = 0
    dx, dy, dz = [0,]*3
    if pbar:
        pbar = tqdm(total=len(xrange)*len(yrange)*len(zrange))
    for ix in xrange:
        for iy in yrange:
            for iz in zrange:
                shifted_aug_ = np.stack([aug_[0]+ix, aug_[1]+iy, aug_[2]+iz])
                overlap_ = src_.shape[1] + aug_.shape[1] - np.unique(np.concatenate([src_, shifted_aug_], axis=1), axis=1).shape[1]
                if pbar:
                    pbar.update(1)
                if overlap_ > max_overlap:
                    max_overlap = overlap_
                    # print(max_overlap)
                    dx = ix
                    dy = iy
                    dz = iz
    if pbar:
        pbar.close()
    return dx, dy, dz

def calculate_offset_dist(src, aug, search_range: list=None, pbar=False):
    xx, yy, zz = np.meshgrid(np.arange(src.shape[0]), np.arange(src.shape[1]), np.arange(src.shape[2]), indexing='ij')

    src_ = np.stack([xx[src==1], yy[src==1], zz[src==1]])
    aug_ = np.stack([xx[aug==1], yy[aug==1], zz[aug==1]])

    if search_range is None:
        xrange = range(
            min(-np.min(aug_[0]), np.min(src_[0])-np.max(aug_[0])), 
            min(src.shape[0]-np.max(aug_[0]), np.max(src_[0])-np.min(aug_[0]))
            )
        yrange = range(
            min(-np.min(aug_[1]), np.min(src_[1])-np.max(aug_[1])), 
            min(src.shape[1]-np.max(aug_[1]), np.max(src_[1])-np.min(aug_[1]))
            )
        zrange = range(
            min(-np.min(aug_[2]), np.min(src_[2])-np.max(aug_[2])), 
            min(src.shape[2]-np.max(aug_[2]), np.max(src_[2])-np.min(aug_[2]))
            )
    else:
        xrange, yrange, zrange = search_range

    max_overlap = 0
    dx, dy, dz = [0,]*3
    if pbar:
        pbar = tqdm(total=len(xrange)*len(yrange)*len(zrange))
    for ix in xrange:
        for iy in yrange:
            for iz in zrange:
                shifted_aug_ = np.stack([aug_[0]+ix, aug_[1]+iy, aug_[2]+iz])
                overlap_ = src_.shape[1] + aug_.shape[1] - np.unique(np.concatenate([src_, shifted_aug_], axis=1), axis=1).shape[1]
                if pbar:
                    pbar.update(1)
                if overlap_ > max_overlap:
                    max_overlap = overlap_
                    # print(max_overlap)
                    dx = ix
                    dy = iy
                    dz = iz
    if pbar:
        pbar.close()
    return dx, dy, dz

def roi_bbox(mask):
    non_zero_indices = np.nonzero(mask)

    # Get minimum and maximum values along each axis
    min_z, max_z = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_x, max_x = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])

    # Return the bounding box coordinates
    bounding_box = [
        min_z,
        max_z,
        min_y,
        max_y,
        min_x,
        max_x,
    ]
    return bounding_box


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
    aug_mask = aug_mask[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    aug = aug[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    aug_mask = crop_to_standard(aug_mask, src_mask.shape)
    aug = crop_to_standard(aug, src.shape)
    # print(np.sum(src_mask), np.sum(aug_mask))

    if np.sum(src_mask) == 0 or np.sum(aug_mask) == 0:
        logging.info('missing organ in src or aug after crop_to_standard')
        return src, src_mask

    dx, dy, dz = calculate_offset_fast(src_mask, aug_mask, pbar=False)

    # remove the src organ
    masked_aug = aug * aug_mask
    shifted_aug = shift(masked_aug,  (dx, dy, dz), order=0, prefilter=False)
    shifted_aug_mask = shift(aug_mask,  (dx, dy, dz), order=0, prefilter=False)
    # inpaint_mask = shifted_aug_mask * src_mask
    inpaint_mask = shifted_aug_mask
    inpaint_src = deepcopy(src)
    inpaint_src[src_mask>0] = 0
    inpaint_src[shifted_aug_mask>0] = shifted_aug[shifted_aug_mask>0]
    # return src
    # inpaint the aug onto the src


    # inpaint_src = src + shifted_aug * inpaint_mask
    return inpaint_src, inpaint_mask

def load_inpaint_model(v):
    ckpt = r'./inpaint_repo/{}'.format(v)
    config = load_yaml(str(Path(ckpt)/'config.yaml'))
    model = UNet(**config['model_config']['network_config'])
    ckpt = torch.load(Path(ckpt)/'inpaint.pth')
    model.load_state_dict(ckpt)
    return model

def load_inpaint_pconv_model(v):
    from inpaint.net import PConvUNet as UNet
    ckpt = r'./inpaint_repo/{}'.format(v)
    config = load_yaml(str(Path(ckpt)/'config.yaml'))
    model = UNet()
    ckpt = torch.load(Path(ckpt)/'inpaint.pth')
    model.load_state_dict(ckpt)
    return model



################################################################################################################################

def debug_inpaint():
    from mostoolkit.io_utils import sitk_load_with_metadata, sitk_save
    src, d, s, o, _ = sitk_load_with_metadata(r'./normal_aug/volume-3-2.nii.gz')
    src_label = sitk_load_with_metadata(r'./normal_aug/labels-3-2.nii.gz')[0]
    aug = sitk_load_with_metadata(r'./normal_aug/volume-3-8.nii.gz')[0]
    aug_label = sitk_load_with_metadata(r'./normal_aug/labels-3-8.nii.gz')[0]
    ret = src
    for i in range(1, 5):
        ret = inpaint_image(ret, aug, src_label, aug_label, i)

    sitk_save(r'./an_aug/test-inpaint.nii.gz', ret, s, o, d)
    sitk_save(r'./an_aug/test-inpaint-seg.nii.gz', src_label, s, o, d)


def debug_offset():
    src_mask = np.zeros([128, 128, 128])
    aug_mask = np.zeros([128, 128, 128])
    # src_mask = np.zeros([80, 80, 80])
    # aug_mask = np.zeros([80, 80, 80])
    # calculate (dx, dy, dz) ==> (-49, -69, -20)
    src_mask[20:40, 20:40, 20:40] = 1
    aug_mask[69:79, 89:99, 30:40] = 1
    # aug_mask[30:50, 20:40, 20:40] = 1
    # dx, dy, dz = calculate_offset(src_mask, aug_mask, pbar=True)
    # print(dx, dy, dz)
    dx, dy, dz = calculate_offset_fast(src_mask, aug_mask, pbar=True)
    print(dx, dy, dz)

def debug_mapping():
    import matplotlib.pyplot as plt
    src_mask = np.zeros([40, 40])
    src_mask[10:20, 10:20] = 1

    aug_mask = np.copy(src_mask)
    aug_mask[10:20, 10:12] = 0

    edt_mask, index_mask = calculate_mapping(src_mask, aug_mask)

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(src_mask)
    axs[1].imshow(aug_mask)
    axs[2].imshow(edt_mask/np.max(edt_mask))
    plt.show()

def debug_dataset():
    from mostoolkit.io_utils import sitk_load_with_metadata
    import matplotlib.pyplot as plt
    root = './data128/full'
    fig, axs = plt.subplots(1,2)
    for f in Path(root).glob('volume*.nii.gz'):
        img = sitk_load_with_metadata(str(f))[0]
        axs[0].imshow(img[64])
        axs[1].imshow(img[:,64])
        # plt.imshow(img[64,:,:])
        axs[0].set_title(f.name)
        print('./data128/full/insight/{}.png'.format(f.name.replace('.nii.gz','')))
        plt.savefig('./data128/full/insight/{}.png'.format(f.name.replace('.nii.gz','')))
        plt.waitforbuttonpress(0.1)

if __name__ == '__main__':
    # debug_inpaint()
    # debug_mapping()
    debug_dataset()
    # debug_offset()