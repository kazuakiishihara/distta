import csv
import datetime as datetime
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import time

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data import datasets, trans
import engine_fun.regisry as regisry
from losses.losses import NCC_vxm, DiceLoss, Grad3d
from utils.utils import AverageMeter, register_model, dice_val_VOI, similarity

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # Number of parameters in millions

def train_model(dataset_label, task, seg_loss, model_label, lr, epochs, batch_size, log_dir, experiment):

    # Write down the training performance into csv file
    csv_path = os.path.join(log_dir, "metrics_per_epoch.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss', 'val_dice', 'val_SSIM_for_x', 'val_SSIM_for_y'])

    # Set the weights of the loss funtion
    weights = [1.0, 1.0, 1.0]

    if not os.path.exists(log_dir+'/model_wts/'):
        print("Creating ckp dir", os.path.abspath(log_dir))
        os.makedirs(log_dir+'/model_wts/')
        ckp_dir = log_dir+'/model_wts/'
    
    # Initialize dataset
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.Seg_norm(dataset_label=dataset_label),
                                        trans.NumpyType((np.float32, np.int16))])
    if dataset_label == "ixi" and task == "ar":
        atlas_dir = 'C:/Users/User/env/DATASETS/IXI/atlas.pkl'
        train_dir = 'C:/Users/User/env/DATASETS/IXI/Train/'
        val_dir = 'C:/Users/User/env/DATASETS/IXI/Val/'
        img_size = (192, 224, 160)
        if seg_loss:
            train_set = datasets.IXIBrainInferDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=val_composed, img_size=img_size)
        else:
            train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed, img_size=img_size)
        val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed, img_size=img_size)
    elif dataset_label == "ixi" and task == "ir":
        train_dir = 'C:/Users/User/env/DATASETS/IXI_ir/Train/'
        val_dir = 'C:/Users/User/env/DATASETS/IXI_ir/Val/'
        img_size = (192, 224, 160)
        if seg_loss:
            train_set = datasets.IXIirInfer(glob.glob(train_dir + '*.pkl'), transforms=val_composed, img_size=img_size)
        else:
            train_set = datasets.IXIir(glob.glob(train_dir + '*.pkl'), transforms=train_composed, img_size=img_size)
        val_set = datasets.IXIirInfer(glob.glob(val_dir + '*.pkl'), transforms=val_composed, img_size=img_size)
    # elif dataset_label == "lpba":
    #     train_dir = 'C:/Users/User/env/DATASETS/LPBA/Train/'
    #     val_dir = 'C:/Users/User/env/DATASETS/LPBA/Val/'
    #     img_size = (160, 192, 160)
    #     train_set = datasets.LPBABrainDatasetS2S(glob.glob(train_dir + '*.pkl'), transforms=train_composed, img_size=img_size)
    #     val_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=val_composed, img_size=img_size)
    else:
        raise ValueError(f"Unsupported dataset_label: {dataset_label}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    model = regisry.build_model(model_label, img_size)
    model.cuda()
    num_params = count_parameters(model)
    print(f"model size: {num_params} million parameters")

    # Initialize spatial transformation function
    reg_model = register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_ncc = NCC_vxm()
    criterion_dsc = DiceLoss()
    criterion_reg = Grad3d(penalty='l2')
    
    best_dsc = 0
    for epoch in range(1, epochs+1):
        '''
        Training phase
        '''
        start_time = time.time()
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch-1, epochs, lr) # Adjust learning rate
            data = [t.cuda() for t in data]
            x, y = data[0], data[1] # x: moving image, y: fixed image
            if seg_loss:
                x_seg, y_seg = data[2], data[3]
            output = model((x, y))

            # Loss calculation
            loss_ncc = criterion_ncc(output[0], y) * weights[0]
            if seg_loss:
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                loss_dsc = criterion_dsc(def_out.long(), y_seg.long()) * weights[1]
                if model_label == 'MoSE':
                    loss_dsc_m = criterion_dsc(output[2][0].long(), x_seg.long()) * 1.0
                    loss_dsc_f = criterion_dsc(output[3][0].long(), y_seg.long()) * 1.0
                    loss_norm_m = torch.norm(output[2][1], p=1)
                    loss_norm_f = torch.norm(output[3][1], p=1)
                    loss_dsc += loss_dsc_m + loss_dsc_f + loss_norm_m + loss_norm_f
            else:
                loss_dsc = 0
            loss_reg = criterion_reg(output[1], y) * weights[2]
            loss = loss_ncc + loss_dsc + loss_reg
            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                idx, len(train_loader), loss.item(), loss_ncc.item(), loss_reg.item()
            ))
        
        training_time = time.time() - start_time
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        metrics = {'train_loss': loss_all.avg, 'training_time' : training_time}
        experiment.log_metrics(metrics, step=epoch)
        
        '''
        Validation phase
        '''
        eval_dsc = AverageMeter()
        eval_ssim_new_y = AverageMeter()  # Track SSIM for y; Deformed vs Fixed
        eval_ssim_new_x = AverageMeter()  # Track SSIM for x; Deformed vs Moving
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x, y = data[0], data[1]
                x_seg, y_seg = data[2], data[3]
                output = model((x, y))
                
                grid_img = mk_grid_img(8, 1, img_size)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])

                dsc = dice_val_VOI(def_out.long(), y_seg.long(), dataset_label=dataset_label) # Dice score
                dsc_new_y = similarity(output[0], y)  # SSIM for y; Deformed vs Fixed
                dsc_new_x = similarity(output[0], x)  # SSIM for x; Deformed vs Moving

                eval_dsc.update(dsc.item(), x.size(0))
                eval_ssim_new_y.update(dsc_new_y.item(), x.size(0))
                eval_ssim_new_x.update(dsc_new_x.item(), x.size(0))

        best_dsc = max(eval_dsc.avg, best_dsc)
        training_time = time.time() - start_time
        remaining_training_time = (training_time * (epochs - epoch)) / 3600
        print('Epoch {} Dice score {:.4f} SSIM Deformed vs Fixed {:.4f} SSIM Deformed vs Moving {:.4f} Time Left {:.2f}h'.format(
            epoch, eval_dsc.avg, eval_ssim_new_y.avg, eval_ssim_new_x.avg, remaining_training_time
        ))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
            }, 
            save_dir=ckp_dir, filename='{}_dsc{:.4f}_epoch{}.pth.tar'.format(model_label, eval_dsc.avg, epoch))
        
        # Write down into csv
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                loss_all.avg,
                eval_dsc.avg,
                eval_ssim_new_x.avg,
                eval_ssim_new_y.avg
            ])
        loss_all.reset()

        # Write down into Comet
        metrics = {
                    'val_dice': eval_dsc.avg,
                    'val_SSIM_for_y': eval_ssim_new_y.avg,
                    'val_SSIM_for_x': eval_ssim_new_x.avg
                    }
        experiment.log_metrics(metrics, step=epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        experiment.log_figure(figure_name="Grid", figure=grid_fig, step=epoch)
        plt.close(grid_fig)
        experiment.log_figure(figure_name="input", figure=x_fig, step=epoch)
        plt.close(x_fig)
        experiment.log_figure(figure_name="ground truth", figure=tar_fig, step=epoch)
        plt.close(tar_fig)
        experiment.log_figure(figure_name="prediction", figure=pred_fig, step=epoch)
        plt.close(pred_fig)
        eval_dsc.reset()
        eval_ssim_new_y.reset()
        eval_ssim_new_x.reset()
        del def_out, def_grid, grid_img, output
    
    experiment.log_parameters({
                                "parameters": num_params,
                                "lr":lr,
                                "scheduler":"cosine",
                                })
    experiment.end()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 56:72, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128, 128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=1):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
