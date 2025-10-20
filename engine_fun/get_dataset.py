import glob
import numpy as np
import os
from torchvision import transforms

from data import datasets, trans

def paired_list(test_dir, label_dir, mask_dir):
    paired_list = []
    files = glob.glob(os.path.join(test_dir, '*.img'))
    for file in files:
        filename = os.path.basename(file)
        atlas_path = os.path.join(label_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        paired_list.append((file, atlas_path, mask_path))
    return paired_list

def get_train_val_dataset(dataset_label, task):

    cfg = dataset_config[f"{dataset_label}_{task}"]

    base_dir = 'C:/Users/User/env/DATASETS/'
    dataset_config = {
        'ixi_ar': {
            'atlas_dir': f'{base_dir}IXI/atlas.pkl',
            'train_dir': f'{base_dir}IXI/Train/',
            'val_dir': f'{base_dir}IXI/Val/',
            'img_size': (192, 224, 160),
            'train_dataset_class': datasets.IXIBrainDataset,
            'val_dataset_class': datasets.IXIBrainInferDataset,
            'args': lambda cfg: [glob.glob(cfg['test_dir'] + '*.pkl')],
        },
        'ixi_ir': {
            'train_dir': f'{base_dir}IXI_ir/Train/',
            'val_dir': f'{base_dir}IXI_ir/Val/',
            'img_size': (192, 224, 160),
            'train_dataset_class': datasets.IXIir,
            'val_dataset_class': datasets.IXIirInfer,
            'args': lambda cfg: [glob.glob(cfg['test_dir'] + '*.pkl')],
        }
    }

    if cfg not in dataset_config:
        raise ValueError(f"Unsupported dataset_label: {cfg}")

    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([
                                    trans.Seg_norm(dataset_label=dataset_label),
                                    trans.NumpyType((np.float32, np.int16))
                                    ])

    train_set = cfg['dataset_class'](*cfg['args'](cfg), transforms=train_composed, img_size=cfg['img_size'])
    val_set = cfg['dataset_class'](*cfg['args'](cfg), transforms=val_composed, img_size=cfg['img_size'])

    return train_set, val_set

def get_test_dataset(dataset_label, task, training_or_inference):
    base_dir = 'C:/Users/User/env/DATASETS/'

    dataset_config = {
        'ixi_ar_training': {
            'test_dir': f'{base_dir}IXI/Test/',
            'img_size': (192, 224, 160),
            'VOI_lbls': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36],
            'dataset_class': datasets.IXIirInfer,
            'args': lambda cfg: [glob.glob(cfg['test_dir'] + '*.pkl')],
        },
        'ixi_ar_training': {
            'test_dir': f'{base_dir}IXI/Test/',
            'img_size': (192, 224, 160),
            'VOI_lbls': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36],
            'dataset_class': datasets.IXIirInfer,
            'args': lambda cfg: [glob.glob(cfg['test_dir'] + '*.pkl')],
        },
        'mgh': {
            'test_dir': f'{base_dir}CLMI/data/MGH10/Heads/',
            'label_dir': f'{base_dir}CLMI/data/MGH10/Atlases/',
            'mask_dir': f'{base_dir}CLMI/data/MGH10/BrainMasks/',
            'img_size': (192, 224, 160),
            'VOI_lbls': list(range(1, 37)),
            'dataset_class': datasets.CLMIirInfer,
            'args': lambda cfg: [paired_list(cfg['test_dir'], cfg['label_dir'], cfg['mask_dir'])],
        },
        'cumc': {
            'test_dir': f'{base_dir}CLMI/data/CUMC12/Heads/',
            'label_dir': f'{base_dir}CLMI/data/CUMC12/Atlases/',
            'mask_dir': f'{base_dir}CLMI/data/CUMC12/BrainMasks/',
            'img_size': (192, 224, 160),
            'VOI_lbls': list(range(1, 27)),
            'dataset_class': datasets.CLMIirInfer,
            'args': lambda cfg: [paired_list(cfg['test_dir'], cfg['label_dir'], cfg['mask_dir'])],
        },
        'ibsr': {
            'test_dir': f'{base_dir}CLMI/data/IBSR18/Heads/',
            'label_dir': f'{base_dir}CLMI/data/IBSR18/Atlases/',
            'mask_dir': f'{base_dir}CLMI/data/IBSR18/BrainMasks/',
            'img_size': (192, 224, 160),
            'VOI_lbls': list(range(1, 31)),
            'dataset_class': datasets.CLMIirInfer,
            'args': lambda cfg: [paired_list(cfg['test_dir'], cfg['label_dir'], cfg['mask_dir'])],
        }
    }

    # 存在チェック
    if dataset_label not in dataset_config:
        raise ValueError(f"Unsupported dataset_label: {dataset_label}")

    cfg = dataset_config[f"{dataset_label}_{task}_{training_or_inference}"]

    # 共通Transform
    test_composed = transforms.Compose([
        trans.Seg_norm(dataset_label=dataset_label),
        trans.NumpyType((np.float32, np.int16))
    ])

    # データセットインスタンス生成
    test_set = cfg['dataset_class'](*cfg['args'](cfg), transforms=test_composed, img_size=cfg['img_size'])

    return test_set, cfg['VOI_lbls']
