import shutil 
import os, sys
from mostoolkit.io_utils import load_json, save_json
from pathlib import Path
from utils import volume_to_label_fname
import random
######################################################################################################
# ct-org dataset
def ctorg_testdata():
    split = load_json(r'./split_train10.json')
    root = Path(r'./data128/full')
    target = Path(r'./nnunet_raw/Dataset001_base10')

    (target / 'imagesTs').mkdir(exist_ok=True)
    (target / 'labelsTs').mkdir(exist_ok=True)

    for n,f in enumerate(split['val']):
        img = root / f
        label = root / f.replace('volume', 'labels')
        id = str(n).zfill(3)
        tar_img = target / 'imagesTs' / 'ctorg_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTs' / 'ctorg_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))
    print('all done')

def ctorg_base():
    split = load_json(r'./split_train10.json')
    root = Path(r'./data128/full')
    target = Path(r'./nnunet_raw/Dataset001_base10')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    (target / 'labelsTr').mkdir(exist_ok=True)
    (target / 'imagesTs').mkdir(exist_ok=True)

    for n,f in enumerate(split['train']):
        img = root / f
        label = root / f.replace('volume', 'labels')
        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'ctorg_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'ctorg_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] = {'background': 0, 'liver': 1, 'bladder': 2, 'lung': 3, 'kidney': 4, 'bones': 5}
    dataset['numTraining'] = 10
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('all done')

def ctorg_anatomix():
    # split = load_json(r'./split_train10.json')
    root = Path(r'./anatomix_final10')
    target = Path(r'./nnunet_raw/Dataset002_atmx10')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)

    (target / 'labelsTr').mkdir(exist_ok=True)
    (target / 'imagesTs').mkdir(exist_ok=True)

    for n,f in enumerate(root.glob('volume*.nii.gz')):
        img = f
        label = root / f.name.replace('volume', 'labels')
        assert f.exists(), 'image {} does not exist'.format(f.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'ctorgatmx_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'ctorgatmx_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] = {'background': 0, 'liver': 1, 'bladder': 2, 'lung': 3, 'kidney': 4, 'bones': 5}
    dataset['numTraining'] = 500
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('all done')

######################################################################################################
# Amos dataset
def amos_test_dataset():
    root = Path(r'./amos128')
    target = Path(r'./nnunet_raw/Dataset001_amos20')
    split = load_json(r'./split_amos_ppoor.json')

    (target / 'imagesTs').mkdir(exist_ok=True, parents=True)
    (target / 'labelsTs').mkdir(exist_ok=True)

    for n,f in enumerate(split['val']):
        img = root / f
        label = root / f.replace('volume', 'labels')
        id = str(n).zfill(3)
        tar_img = target / 'imagesTs' / 'amos_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTs' / 'amos_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))
    print('amos testdataset all done')

def amos_base40_dataset():
    root = Path(r'./amos128')
    target = Path(r'./nnunet_raw/Dataset001_amos40')
    split = load_json(r'./split_amos_poor.json')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    
    (target / 'labelsTr').mkdir(exist_ok=True)
    (target / 'imagesTs').mkdir(exist_ok=True)

    for n,f in enumerate(split['train']):
        img = root / f.replace('amos', 'volume')
        label = root / f.replace('amos', 'labels')
        assert img.exists(), 'image {} does not exist'.format(img.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'amos_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'amos_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] ={'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gall bladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'arota': 8, 'postcava': 9, 'pancreas': 10, 'right adrenal gland': 11, 'left adrenal gland': 12, 'duodenum': 13, 'bladder': 14, 'prostate/uterus': 15}
    dataset['numTraining'] = 40
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done'.format(target.name))
    # print('all done')

def amos_base20_dataset():
    root = Path(r'./amos128')
    target = Path(r'./nnunet_raw/Dataset001_amos20')
    split = load_json(r'./split_amos_ppoor.json')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    
    (target / 'labelsTr').mkdir(exist_ok=True)
    # (target / 'imagesTs').mkdir(exist_ok=True)

    # n_aug = 200

    # for n in range(n_aug):
    #     f = random.choice(split['train'])
    for n,f in enumerate(split['train']):
        img = root / f
        label = root / f.replace('volume', 'labels')
        assert img.exists(), 'image {} does not exist'.format(img.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'amos_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'amos_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] ={'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gall bladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'arota': 8, 'postcava': 9, 'pancreas': 10, 'right adrenal gland': 11, 'left adrenal gland': 12, 'duodenum': 13, 'bladder': 14, 'prostate/uterus': 15}
    dataset['numTraining'] = 20
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done'.format(target.name))
    # print('all done')

def amos_ppoor_reduced_organ_dataset():
    root = Path(r'./amos20atmx_ro')
    target = Path(r'./nnunet_raw/Dataset004_amos20atmxRO')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    
    (target / 'labelsTr').mkdir(exist_ok=True)
    # (target / 'imagesTs').mkdir(exist_ok=True)

    for n,f in enumerate(root.glob('volume*.nii.gz')):
        img = f
        label = Path(volume_to_label_fname(f))
        assert f.exists(), 'image {} does not exist'.format(f.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'amos_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'amos_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] ={'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gall bladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'arota': 8, 'postcava': 9, 'pancreas': 10, 'right adrenal gland': 11, 'left adrenal gland': 12, 'duodenum': 13, 'bladder': 14, 'prostate/uterus': 15}
    dataset['numTraining'] = 200
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done'.format(target.name))
    # print('all done')

def amos_atmx_v2_dataset():
    # root = Path(r'./amos20atmxV2')
    root = Path(r'./amos20atmxV2RO')
    # target = Path(r'./nnunet_raw/Dataset003_amos20atmxV2')
    target = Path(r'./nnunet_raw/Dataset002_amos20atmxV2RO')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    
    (target / 'labelsTr').mkdir(exist_ok=True)
    # (target / 'imagesTs').mkdir(exist_ok=True)

    for n,f in enumerate(root.glob('volume*.nii.gz')):
        img = f
        label = root / f.name.replace('volume', 'labels')
        assert f.exists(), 'image {} does not exist'.format(f.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'amos_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'amos_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] ={'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gall bladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'arota': 8, 'postcava': 9, 'pancreas': 10, 'right adrenal gland': 11, 'left adrenal gland': 12, 'duodenum': 13, 'bladder': 14, 'prostate/uterus': 15}
    dataset['numTraining'] = 200
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done'.format(target.name))
    # print('all done')

def amos_rich_dataset():
    root = Path(r'./amos_rich_anatomix')
    target = Path(r'./nnunet_raw/Dataset004_amos200atmx')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    
    (target / 'labelsTr').mkdir(exist_ok=True)
    (target / 'imagesTs').mkdir(exist_ok=True)

    for n,f in enumerate(root.glob('volume*.nii.gz')):
        img = f
        label = root / f.name.replace('volume', 'labels')
        assert f.exists(), 'image {} does not exist'.format(f.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'amosr_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'amosr_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT'}
    dataset['labels'] = {'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gall bladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'arota': 8, 'postcava': 9, 'pancreas': 10, 'right adrenal gland': 11, 'left adrenal gland': 12, 'duodenum': 13, 'bladder': 14, 'prostate/uterus': 15}
    dataset['numTraining'] = 500
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done'.format(target.name))
    # print('all done')


def amos_experiment_dataset(exp='cutmix', data_root=None, clean=False, length=-1, nnunet_raw=None):
    # target = {
    #     'cutmix': 'Dataset010_amoscutmix',
    #     'carvemix': 'Dataset011_amoscarvemix',
    #     'objectaug': 'Dataset012_amosobjectaug',
    #     'atmxcutmix': 'Dataset013_amosatmxcutmix',
    #     'atmxv3': 'Dataset014_amosatmxv3',
    #     # 'anatomix': 'Dataset006_dectatmx',
    #     'atmx': 'Dataset002_amos20atmxV2RO',
    # }[exp]
    target = {
        'cutmix': 'Dataset002_amoscutmix',
        'carvemix': 'Dataset003_amoscarvemix',
        'objectaug': 'Dataset004_amosobjectaug',
        'anatomix': 'Dataset005_amos20atmx',
    }[exp]
    if nnunet_raw is None:
        target = Path(r'./nnunet_raw') / target
    else:
        target = Path(nnunet_raw) / target
    target.mkdir(exist_ok=True, parents=True)
    if data_root is None:
        objs = Path(r'./amos20{}'.format(exp)).glob('volume*.nii.gz')
    else:
        objs = Path(data_root).glob('volume*.nii.gz')
    objs = list(objs)
    if length > 0:
        objs = objs[:length]
    print(target, len(objs))

    if clean:
        shutil.rmtree(str(target))

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    (target / 'labelsTr').mkdir(exist_ok=True)
    for n, img in enumerate(objs):
        label = volume_to_label_fname(img)
        id = str(n).zfill(3)
        tar_img = target / 'imagesTr' / 'amos_{}_0000.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'amos_{}.nii.gz'.format(id)

        shutil.copy(str(img), str(tar_img))
        # shutil.copy(str(imgL), str(tar_imgL))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))
    
    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT',}
    dataset['labels'] = {'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gall bladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'arota': 8, 'postcava': 9, 'pancreas': 10, 'right adrenal gland': 11, 'left adrenal gland': 12, 'duodenum': 13, 'bladder': 14, 'prostate/uterus': 15}
    dataset['numTraining'] = len(objs)
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done, dataset len: {}'.format(target.name, dataset['numTraining']))


##################################################################################################################################
# DECT datasets 

def dect_base_dataset():
    root = Path(r'./dect128')
    target = Path(r'./nnunet_raw/Dataset005_dect')
    split = load_json(r'./split_dect.json')

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)    
    (target / 'labelsTr').mkdir(exist_ok=True)

    for n,id in enumerate(split['train']):
        imgH = root / f'volume_{id}_1.nii.gz'
        imgL = root / f'volume_{id}_2.nii.gz'
        label = root / f'labels_{id}.nii.gz'
        assert imgH.exists(), 'image {} does not exist'.format(imgH.name)
        assert imgL.exists(), 'image {} does not exist'.format(imgL.name)
        assert label.exists(), 'label {} does not exist'.format(label.name)

        id = str(n).zfill(3)
        tar_imgH = target / 'imagesTr' / 'dect_{}_0000.nii.gz'.format(id)
        tar_imgL = target / 'imagesTr' / 'dect_{}_0001.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'dect_{}.nii.gz'.format(id)

        shutil.copy(str(imgH), str(tar_imgH))
        shutil.copy(str(imgL), str(tar_imgL))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(id))

    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT', '1': 'CT'}
    dataset['labels'] ={'background': 0, 'lkidney': 1, 'rkidney': 2, 'liver': 3, 'spleen': 4, 'llung': 5, 'rlung': 6, 'pancreas': 7, 'gbladder': 8, 'aorta': 9, }
    dataset['numTraining'] = 22
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done'.format(target.name))


def dect_test_dataset():
    root = Path(r'./dect128')
    target = Path(r'./nnunet_raw/Dataset005_dect')
    split = load_json(r'./split_dect_he.json')

    (target / 'imagesTs').mkdir(exist_ok=True)
    (target / 'labelsTs').mkdir(exist_ok=True)

    for n,f in enumerate(split['val']):
        imgH = root / f
        imgL = root / f.replace('_1.nii.gz', '_2.nii.gz')
        label = volume_to_label_fname(imgH, True)
        id = str(n).zfill(3)
        tar_imgH = target / 'imagesTs' / 'dect_{}_0000.nii.gz'.format(id)
        tar_imgL = target / 'imagesTs' / 'dect_{}_0001.nii.gz'.format(id)
        tar_label = target / 'labelsTs' / 'dect_{}.nii.gz'.format(id)

        shutil.copy(str(imgH), str(tar_imgH))
        shutil.copy(str(imgL), str(tar_imgL))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(id))
    print('dect testdataset all done')


def dect_experiment_dataset(exp='cutmix', data_root=None, length=-1, clean=False, nnunet_raw=None):
    target = {
        'cutmix': 'Dataset007_dectcutmix',
        'carvemix': 'Dataset008_dectcarvemix',
        'objectaug': 'Dataset009_dectobjectaug',
        'anatomix': 'Dataset010_dectatmx',
        # 'atmx': 'Dataset006_dectatmx',
    }[exp]
    if nnunet_raw is None:
        target = Path(r'./nnunet_raw') / target
    else:
        target = Path(nnunet_raw) / target
    if clean and target.exists():
        shutil.rmtree(target)
    target.mkdir(exist_ok=True, parents=True)

    
    if data_root is None:
        data_root_he = Path(r'./dect_he_{}'.format(exp))
        data_root_le = Path(r'./dect_le_{}'.format(exp))
    else:
        data_root_he = Path(data_root[0])
        data_root_le = Path(data_root[1])

    objs = data_root_he.glob('volume*.nii.gz')
    objs = list(objs)
    if length > 0:
        objs = objs[:length]
    print(target, len(objs))

    (target / 'imagesTr').mkdir(exist_ok=True, parents=True)
    (target / 'labelsTr').mkdir(exist_ok=True)
    for n, img in enumerate(objs):
        imgH = img
        imgL = data_root_le / img.name.replace('_1_', '_2_')
        if imgL.exists() is False:
            continue
        label = volume_to_label_fname(img, multimodality=True)
        id = str(n).zfill(3)
        tar_imgH = target / 'imagesTr' / 'dect_{}_0000.nii.gz'.format(id)
        tar_imgL = target / 'imagesTr' / 'dect_{}_0001.nii.gz'.format(id)
        tar_label = target / 'labelsTr' / 'dect_{}.nii.gz'.format(id)
        # print(imgH, imgL, tar_imgH, tar_imgL)

        shutil.copy(str(imgH), str(tar_imgH))
        shutil.copy(str(imgL), str(tar_imgL))
        shutil.copy(str(label), str(tar_label))
        print('{} done'.format(img))
        # break
    
    # create the dataset.json 
    dataset = dict()
    dataset['channel_names'] = {'0': 'CT', '1': 'CT'}
    dataset['labels'] = {'background': 0, 'lkidney': 1, 'rkidney': 2, 'liver': 3, 'spleen': 4, 'llung': 5, 'rlung': 6, 'pancreas': 7, 'gbladder': 8, 'aorta': 9, }
    dataset['numTraining'] = len(list(objs))
    dataset['file_ending'] = '.nii.gz'
    dataset_fname = (target/'dataset.json')
    save_json(str(dataset_fname), dataset)
    print('{} all done, dataset len: {}'.format(target.name, dataset['numTraining']))


if __name__ == '__main__':
    # ctorg_testdata()
    # amos_base20_dataset()
    # amos_atmx_v2_dataset()
    # amos_ppoor_reduced_organ_dataset()
    # amos_test_dataset()
    # amos_base200_dataset()
    # amos_test_dataset()
    # amos_ppoor_dataset(r'./amospp')
    # amos_rich_dataset()
    # dect_base_dataset()
    # dect_test_dataset()
    # dect_experiment_dataset('cutmix')
    # dect_experiment_dataset('objectaug')
    # dect_experiment_dataset('carvemix')
    # dect_experiment_dataset('atmx')

    # amos_experiment_dataset('cutmix', clean=True)
    # amos_experiment_dataset('atmx', data_root='./amos20atmxV2RO', clean=True)
    # amos_experiment_dataset('atmxcutmix', data_root='./amos_atmx_cutmix', clean=True)
    # amos_experiment_dataset('atmxv3', data_root='./amos_atmx_cutmix', length=200)
    # amos_experiment_dataset('carvemix')
    # amos_experiment_dataset('objectaug')
    
    l = 200
    # l = 500
    # l = 1000
    nnunet_raw = r'nnunet_raw_t10'
    # nnunet_raw = r'./nnunet_raw'
    # nnunet_raw = r'./nnunet_raw_t50'

    # amos_experiment_dataset(exp='cutmix', data_root='amoscutmix1k', length=l, nnunet_raw=nnunet_raw)
    # amos_experiment_dataset(exp='carvemix', data_root='amoscarvemix1k', length=l, nnunet_raw=nnunet_raw)
    amos_experiment_dataset(exp='objectaug', data_root='amosobjectaug1k', length=l, nnunet_raw=nnunet_raw)
    # amos_experiment_dataset(exp='anatomix', data_root='amosatmx1k', length=l, nnunet_raw=nnunet_raw)

    # dect_experiment_dataset(exp='cutmix', data_root=['decthecutmix1k', 'dectlecutmix1k'], length=l, nnunet_raw=nnunet_raw)
    dect_experiment_dataset(exp='objectaug', data_root=['dectheobjectaug1k', 'dectleobjectaug1k'], length=l, nnunet_raw=nnunet_raw)
    # dect_experiment_dataset(exp='carvemix', data_root=['decthecarvemix1k', 'dectlecarvemix1k'], length=l, nnunet_raw=nnunet_raw)
    # dect_experiment_dataset(exp='anatomix', data_root=['dectheatmx1k', 'dectleatmx1k'], length=l, nnunet_raw=nnunet_raw)
# 