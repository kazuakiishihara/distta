import argparse
from comet_ml import Experiment
import numpy as np
import random
import torch

from engine_fun.engine_tta import adapt_model

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=True

same_seeds(121)

def parse_args():
    parser = argparse.ArgumentParser(description='Test Model for Image Registration')

    # Dataset setting
    parser.add_argument('--dataset_label', type=str, default='ixi', help='Dataset label (ixi, lpba)')

    # Task setting
    parser.add_argument('--task', type=str, default='ar', help='Task label (ar: atlas-based registration, ir: inter-patient registration)')
    
    # Training setting
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('-e', "--epochs", type=int, default=50)
    parser.add_argument('-bs', "--batch_size", type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # Create the Comet ML experiment instance.
    project_name = args.dataset_label + "_" + args.task
    experiment = Experiment(
                            api_key="1qdOUPyuagq2Kt5EoyTJu1ZQa",
                            project_name=project_name,
                            workspace="k-ishihara",
                            )
    experiment.set_name('Nov07-220129_TransMorph_DS_TTA')

    log_dir='./logs/ixi_ar/Nov07-220129_TransMorph_DS/'

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

    adapt_model(
        dataset_label=args.dataset_label,
        task=args.task,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_dir=log_dir,
        experiment=experiment,
    )
