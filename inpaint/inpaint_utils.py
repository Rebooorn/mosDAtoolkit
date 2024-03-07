from mostoolkit.io_utils import sitk_save

'''
The code is from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv \
The original code is 2D, here it is modified as 3D.
'''

import argparse
import numpy as np
import random
from PIL import Image
import os

def parse_trainer_args(args: dict):
    # os.name == 'nt' ==> Windows ==> debug
    if os.name == 'nt':
        args['strategy'] = 'auto'
    else:
        args['strategy'] = 'ddp'
        args['enable_progress_bar'] = False
    return args


# action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
action_list = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]

def random_walk(canvas, ini_z, ini_x, ini_y, length):
    z = ini_z
    x = ini_x
    y = ini_y
    img_size = canvas.shape
    z_list = []
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        z = np.clip(z + action_list[r][0], a_min=0, a_max=img_size[0] - 1)
        x = np.clip(x + action_list[r][1], a_min=0, a_max=img_size[1] - 1)
        y = np.clip(y + action_list[r][2], a_min=0, a_max=img_size[2] - 1)
        z_list.append(z)
        x_list.append(x)
        y_list.append(y)

    canvas[np.array(z_list), np.array(x_list), np.array(y_list)] = 0
    return canvas

def worker_fn(i, new_nslices, new_spatial):
    canvas = np.ones((new_nslices, new_spatial, new_spatial)).astype("i")
    ini_x = random.randint(0, new_spatial - 1)
    ini_y = random.randint(0, new_spatial - 1)
    ini_z = random.randint(0, new_nslices-1)
    mask = random_walk(canvas, ini_z, ini_x, ini_y, (new_spatial//2)**3)
    print("save:", i, np.sum(mask))
    # print(mask.shape)

    # img = Image.fromarray(mask * 255).convert('1')
    sitk_save('masks/{:06d}.nii.gz'.format(i), mask, [3.5, 3.5, 3.5], [0,0,0], [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])

if __name__ == '__main__':
    import os
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='masks')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    new_space = 3.5     # 128
    new_spatial = 128
    new_nslices = 128

    args = [(i, new_nslices, new_spatial) for i in range(args.N)]
    with Pool(processes=32) as pool:
        ret = pool.starmap(worker_fn, args)


    # for i in range(args.N):
    #     canvas = np.ones((new_nslices, new_spatial, new_spatial)).astype("i")
    #     ini_x = random.randint(0, new_spatial - 1)
    #     ini_y = random.randint(0, new_spatial - 1)
    #     ini_z = random.randint(0, new_nslices-1)
    #     mask = random_walk(canvas, ini_z, ini_x, ini_y, (new_spatial//2)**3)
    #     print("save:", i, np.sum(mask))
    #     # print(mask.shape)

    #     # img = Image.fromarray(mask * 255).convert('1')
    #     sitk_save('{:s}/{:06d}.nii.gz'.format(args.save_dir, i), mask, [3.5, 3.5, 3.5], [0,0,0], [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
        # break
        