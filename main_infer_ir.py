import argparse
import glob
from natsort import natsorted
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import datasets, trans
import engine_fun.regisry as regisry 
from utils.utils import AverageMeter, register_model, jacobian_determinant_vxm, dice_val_VOI
from utils.metrics import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test Model for Image Registration')
    parser.add_argument('--dataset_label', type=str, default='ixi', help='Dataset label (ixi, lpba)')
    return parser.parse_args()

def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    # return np.sum(Ja_value<0)
    return np.sum(Ja_value < 0) / Ja_value.size * 100

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def paired_list(test_dir, label_dir, mask_dir):
    paired_list = []
    files = glob.glob(os.path.join(test_dir, '*.img'))
    for file in files:
        filename = os.path.basename(file)
        atlas_path = os.path.join(label_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        paired_list.append((file, atlas_path, mask_path))
    return paired_list

def main(dataset_label):

    save_dir = './Quantitative_Results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    csv_writter('Model, DSC, Affine DSC, HD95, NJD, %|J|=<0, Params(M)', save_dir+'/{}_ir_performance_results'.format(dataset_label))

    model_idx = -1
    project_name = "ixi_ir"
    log_dir = './logs/' + project_name + '/'
    experiments = [run_id for run_id in os.listdir(log_dir)]

    if dataset_label == 'ixi':
        test_dir = 'C:/Users/User/env/DATASETS/IXI_ir/Test/'
        img_size = (192, 224, 160)
        VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
        test_composed = transforms.Compose([trans.Seg_norm(dataset_label=dataset_label),
                                            trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.IXIirInfer(glob.glob(test_dir + '*.pkl'), transforms=test_composed, img_size=img_size)
    # elif dataset_label == 'lpba':
    #     test_dir = 'C:/Users/User/env/DATASETS/LPBA/Test/'
    #     img_size = (192, 224, 160)
    #     VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #                 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    #                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    #                 48, 49, 50, 51, 52, 53, 54]
    #     test_composed = transforms.Compose([trans.Seg_norm(dataset_label=dataset_label),
    #                                         trans.NumpyType((np.float32, np.int16))])
    #     test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(test_dir + '*.pkl'), transforms=test_composed, img_size=img_size)
    elif dataset_label == 'mgh':
        test_dir = 'C:/Users/User/env/DATASETS/CLMI/data/MGH10/Heads/'
        label_dir = 'C:/Users/User/env/DATASETS/CLMI/data/MGH10/Atlases/'
        mask_dir = 'C:/Users/User/env/DATASETS/CLMI/data/MGH10/BrainMasks/'
        img_size = (192, 224, 160)
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        test_composed = transforms.Compose([trans.Seg_norm(dataset_label=dataset_label),
                                            trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.CLMIirInfer(paired_list(test_dir, label_dir, mask_dir), transforms=test_composed, img_size=img_size)
    elif dataset_label == 'cumc':
        test_dir = 'C:/Users/User/env/DATASETS/CLMI/data/CUMC12/Heads/'
        label_dir = 'C:/Users/User/env/DATASETS/CLMI/data/CUMC12/Atlases/'
        mask_dir = 'C:/Users/User/env/DATASETS/CLMI/data/CUMC12/BrainMasks/'
        img_size = (192, 224, 160)
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        test_composed = transforms.Compose([trans.Seg_norm(dataset_label=dataset_label),
                                            trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.CLMIirInfer(paired_list(test_dir, label_dir, mask_dir), transforms=test_composed, img_size=img_size)
    elif dataset_label == 'ibsr':
        test_dir = 'C:/Users/User/env/DATASETS/CLMI/data/IBSR18/Heads/'
        label_dir = 'C:/Users/User/env/DATASETS/CLMI/data/IBSR18/Atlases/'
        mask_dir = 'C:/Users/User/env/DATASETS/CLMI/data/IBSR18/BrainMasks/'
        img_size = (192, 224, 160)
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        test_composed = transforms.Compose([trans.Seg_norm(dataset_label=dataset_label),
                                            trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.CLMIirInfer(paired_list(test_dir, label_dir, mask_dir), transforms=test_composed, img_size=img_size)
    else:
        raise ValueError(f"Unsupported dataset_label: {dataset_label}")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    for run_id in experiments:
        model_label = run_id.split("_")[1]
        model_dir = log_dir + run_id + '/model_wts/'
        print('Run id: {}'.format(run_id))

        model = regisry.build_model(model_label, img_size)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], weights_only=False)['state_dict']
        print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
        model.load_state_dict(best_model)
        model.cuda()
        num_params = count_parameters(model)
        reg_model = register_model(img_size, 'nearest')
        reg_model.cuda()

        eval_dsc_def = AverageMeter()
        eval_dsc_raw = AverageMeter()
        eval_det = AverageMeter()
        eval_njd = AverageMeter()
        eval_hd95 = AverageMeter()
        with torch.no_grad():
            stdy_idx = 0
            for data in test_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x, y = data[0], data[1]
                x_seg, y_seg = data[2], data[3]
                output = model((x, y))

                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]

                # Dice Similarity Coefficients (DSC)
                dsc_trans = dice_val_VOI(def_out.long(), y_seg.long(), dataset_label=dataset_label)
                dsc_raw = dice_val_VOI(x_seg.long(), y_seg.long(), dataset_label=dataset_label)

                # 95% Maximum Hausdorff Distance (HD95)
                hd95 = 0
                count = 0
                for i in VOI_lbls:
                    if ((x_seg==i).sum()==0) or ((y_seg==i).sum()==0):
                        print("contunue")
                        continue
                    hd95 += compute_robust_hausdorff(
                                compute_surface_distances(
                                                        (y_seg.long().detach().cpu().numpy()[0, 0, ...]==i), 
                                                        (def_out.long().detach().cpu().numpy()[0, 0, ...]==i), 
                                                        np.ones(3)
                                                        ), 95.)
                    count += 1
                hd95 /= count

                # Percentage of Nagative Jacobian DFeterminants (NJD)
                flow = output[1].detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()
                NJD_val = NJD(flow)

                # Percentage of Negative Values of the Jacobian Determinant (%|J|<=0))
                jac_det = jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :, :])
                folding_ratio = np.sum(jac_det <= 0) / np.prod(tar.shape) * 100

                print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
                eval_dsc_def.update(dsc_trans.item(), x.size(0))
                eval_dsc_raw.update(dsc_raw.item(), x.size(0))
                eval_hd95.update(hd95.item(), x.size(0))
                eval_njd.update(NJD_val.item(), x.size(0))
                eval_det.update(folding_ratio, x.size(0))
                stdy_idx += 1
            
            print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                        eval_dsc_def.std,
                                                                                        eval_dsc_raw.avg,
                                                                                        eval_dsc_raw.std))
            print('deformed det: {:.3f}, std: {:.3f}'.format(eval_det.avg, eval_det.std))
            line = "{}, {:.6f}+/-{:.6f}, {:.6f}+/-{:.6f}, {:.6f}+/-{:.6f}, {:.8f}+/-{:.8f}, {:.8f}+/-{:.8f}, {:.4f}".format(
                                                                                run_id.split("_", 1)[1],
                                                                                eval_dsc_def.avg, eval_dsc_def.std,
                                                                                eval_dsc_raw.avg, eval_dsc_raw.std,
                                                                                eval_hd95.avg, eval_hd95.std,
                                                                                eval_njd.avg, eval_njd.std,
                                                                                eval_det.avg, eval_det.std,
                                                                                num_params
                                                                                )
            csv_writter(line, save_dir+'/{}_ir_performance_results'.format(dataset_label))


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

    main(dataset_label=args.dataset_label)
