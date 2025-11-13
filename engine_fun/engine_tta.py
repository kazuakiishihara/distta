import glob
import matplotlib.pyplot as plt
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
from utils.utils import AverageMeter, register_model

def adapt_model(dataset_label, task, lr, epochs, batch_size, log_dir, experiment):

    # Set the weights of the loss funtion
    weights = [1.0, 1.0]

    model_dir = log_dir + '/model_wts/'
    pretrained_path = model_dir + natsorted(os.listdir(model_dir))[-1]
    if not os.path.exists(log_dir+'tta/'+dataset_label+'/model_wts/'):
        print("Creating ckp dir", os.path.abspath(log_dir))
        os.makedirs(log_dir+'tta/'+dataset_label+'/model_wts/')
    ckp_dir = log_dir+'tta/'+dataset_label+'/model_wts/'

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

    # Initialize spatial transformation function
    reg_model = register_model(img_size, 'nearest')
    reg_model.cuda()

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

        # Write down to Comet
        plt.switch_backend('agg')
        y_fig = comput_fig_flow(y)
        perturb_fig = comput_fig_flow(output[2][0])
        adapted_y_fig = comput_fig_flow(output[2][1])
        experiment.log_figure(figure_name="Fixed Image", figure=y_fig, step=epoch)
        plt.close(y_fig)
        experiment.log_figure(figure_name="Perturbation in ID", figure=perturb_fig, step=epoch)
        plt.close(perturb_fig)
        experiment.log_figure(figure_name="Altered Fixed Image", figure=adapted_y_fig, step=epoch)
        plt.close(adapted_y_fig)
        del def_out, output
        
    experiment.end()

def comput_fig_flow(img):
    b, c, d, h, w = img.shape
    indices = np.linspace(0, d - 1, 16, dtype=int)
    img = img.detach().cpu().numpy()[0, 0]
    img = img[indices, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

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
