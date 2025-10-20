import argparse
import glob
from natsort import natsorted
import numpy as np
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data import datasets, trans
from losses.losses import NCC_vxm, Grad3d
from test_time_adaptation.prompt import TransMorph_SPTTA
from utils.utils import AverageMeter

def parse_args():
    parser = argparse.ArgumentParser(description='Test Model for Image Registration')

    # Dataset setting
    parser.add_argument('--dataset_label', type=str, default='ixi', help='Dataset label (ixi, lpba)')

    # Task setting
    parser.add_argument('--task', type=str, default='ar', help='Task label (ar: atlas-based registration, ir: inter-patient registration)')
    
    # Training setting
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('-e', "--epochs", type=int, default=10)
    parser.add_argument('-bs', "--batch_size", type=int, default=1)

    return parser.parse_args()

def main(dataset_label='ixi', task='ar', lr=0.0001, epochs=50, batch_size=1, log_dir='./logs/ixi_ar/Oct14-205009_TransMorph/'):

    # Set the weights of the loss funtion
    weights = [1.0, 1.0]

    model_dir = log_dir + '/model_wts/'
    pretrained_path = model_dir + natsorted(os.listdir(model_dir))[-1]
    if not os.path.exists(log_dir+'/model_wts_tta/'):
        print("Creating ckp dir", os.path.abspath(log_dir))
        os.makedirs(log_dir+'/model_wts_tta/')
    ckp_dir = log_dir+'/model_wts_tta/'

    if dataset_label == 'ixi':
        atlas_dir = 'C:/Users/User/env/DATASETS/IXI/atlas.pkl'
        test_dir = 'C:/Users/User/env/DATASETS/IXI/Test/'
        img_size = (192, 224, 160)
        test_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        test_set = datasets.IXIBrainDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed, img_size=img_size)
    elif dataset_label == 'mgh':
        atlas_dir = 'C:/Users/User/env/DATASETS/IXI/atlas.pkl'
        test_dir = 'C:/Users/User/env/DATASETS/CLMI/data/MGH10/Heads/'
        label_dir = 'C:/Users/User/env/DATASETS/CLMI/data/MGH10/Atlases/'
        mask_dir = 'C:/Users/User/env/DATASETS/CLMI/data/MGH10/BrainMasks/'
        img_size = (192, 224, 160)
        test_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        test_set = datasets.CLMIar(paired_list(test_dir, label_dir, mask_dir), atlas_dir, transforms=test_composed, img_size=img_size)
    else:
        raise ValueError(f"Unsupported dataset_label: {dataset_label}")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = TransMorph_SPTTA(pretrained_path)
    model.cuda()
    for k, v in model.named_parameters():
        if v.requires_grad:
            print('Trainable', k)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty='l2')

    for epoch in range(1, epochs+1):
        '''
        Adaptation phase
        '''
        start_time = time.time()
        loss_all = AverageMeter()
        idx = 0
        for data in test_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch-1, epochs, lr) # Adjust learning rate
            data = [t.cuda() for t in data]
            x, y = data[0], data[1] # x: moving image, y: fixed image
            output = model((x, y))
            loss_ncc = criterion_ncc(output[0], y) * weights[0]
            loss_reg = criterion_reg(output[1], y) * weights[1]
            loss = loss_ncc  + loss_reg
            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                idx, len(test_loader), loss.item(), loss_ncc.item(), loss_reg.item()
            ))
        
        adaptation_time = time.time() - start_time
        remaining_adaptation_time = (adaptation_time * (epochs - epoch)) / 3600
        print('Epoch {} loss {:.4f} Time Left {:.2f}h'.format(
            epoch, loss_all.avg, remaining_adaptation_time
        ))

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, 
            save_dir=ckp_dir, filename='{}_epoch{}.pth.tar'.format('TransMorph', epoch))
        loss_all.reset()

def paired_list(test_dir, label_dir, mask_dir):
    paired_list = []
    files = glob.glob(os.path.join(test_dir, '*.img'))
    for file in files:
        filename = os.path.basename(file)
        atlas_path = os.path.join(label_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        paired_list.append((file, atlas_path, mask_path))
    return paired_list

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=1):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

if __name__ == '__main__':

    args = parse_args()

    # GPU configuration
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU:', GPU_num)
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using:', torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available?', GPU_avai)

    main(
        dataset_label=args.dataset_label,
        task=args.task,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
