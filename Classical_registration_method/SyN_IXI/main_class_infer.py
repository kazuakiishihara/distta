import glob
import os, utils, torch
import sys, ants
from torch.utils.data import DataLoader
from data_IXI import datasets, trans
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import nibabel as nib
import torch.nn as nn

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def main():
    atlas_dir = 'C:/Users/User/env/DATASETS/IXI/atlas.pkl'
    test_dir = 'C:/Users/User/env/DATASETS/IXI/Test/'
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    # dict = utils.process_label()
    # line = ''
    # for i in range(46):
    #     line = line + ',' + dict[i]
    # csv_writter(line+','+'non_jec', 'ants_IXI')
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    stdy_idx = 0
    eval_dsc_def = AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()
            x = ants.from_numpy(x)
            y = ants.from_numpy(y)

            x_ants = ants.from_numpy(x_seg.astype(np.float32))
            y_ants = ants.from_numpy(y_seg.astype(np.float32))

            reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            def_seg = ants.apply_transforms(fixed=y_ants,
                                            moving=x_ants,
                                            transformlist=reg12['fwdtransforms'],
                                            interpolator='nearestNeighbor',)
            
            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2) # flow: (3, 192, 224, 160)
            print('flow: ' + str(flow.shape))
            def_seg = def_seg.numpy()
            def_seg = torch.from_numpy(def_seg[None, None, ...]) # def_seg: torch.Size([1, 1, 192, 224, 160])
            y_seg = torch.from_numpy(y_seg[None, None, ...])     # y_seg  : torch.Size([1, 1, 192, 224, 160])
            print('def_seg: ' + str(def_seg.shape))
            print('y_seg: ' + str(y_seg.shape))

            dsc_trans = utils.dice_val_VOI(def_seg.long(), y_seg.long(), dataset_label="ixi")
            # dsc_trans = utils.dice_val(def_seg.long(), y_seg.long(), 46)
            eval_dsc_def.update(dsc_trans.item(), 1)
            jac_det = utils.jacobian_determinant_vxm(flow)
            # line = utils.dice_val_substruct(def_seg.long(), y_seg.long(), stdy_idx)
            # line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            # csv_writter(line, 'SyN')
            print('DSC: {:.4f}'.format(dsc_trans.item()))

        print('Deformed DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    main()
