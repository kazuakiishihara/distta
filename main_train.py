import argparse
from comet_ml import Experiment
import datetime as datetime
import numpy as np
import os
import random
import sys
import torch

from engine_fun.engine_train import train_model

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

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train Model for Image Registration')

    # Dataset setting
    parser.add_argument('--dataset_label', type=str, default='ixi', help='Dataset label (ixi, lpba)')

    # Task setting
    parser.add_argument('--task', type=str, default='ar', help='Task label (ar: atlas-based registration, ir: inter-patient registration)')
    parser.add_argument('--segloss', type=bool, default=False, help='Task label (ar: atlas-based registration, ir: inter-patient registration)')
    
    # Training setting
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('-e', "--epochs", type=int, default=30)
    parser.add_argument('-bs', "--batch_size", type=int, default=1)

    # Network structure settings
    parser.add_argument('--model_label', type=str, default='TransMorph', 
                        help='Model label (TransMorph, IIRPNet, EfficientMorph)')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # Generate run_id and project name
    dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))) # HK time
    run_id = '_'.join([dt.strftime('%b%d-%H%M%S'),  # time
                        args.model_label            # model label
                        ])
    project_name = args.dataset_label + "_" + args.task

    # Create the Comet ML experiment instance.
    # Important: this should be done only in the main process to avoid creating multiple experiments when using num_workers > 0.
    experiment = Experiment(
                            api_key="1qdOUPyuagq2Kt5EoyTJu1ZQa",
                            project_name=project_name,
                            workspace="k-ishihara",
                            )
    experiment.set_name(run_id)
    experiment.add_tag(args.model_label)

    # Mkdirs for log directory
    log_dir = os.path.join('./logs', project_name, run_id)
    os.path.exists(log_dir) or os.makedirs(log_dir)

    sys.stdout = Logger(log_dir+'/')
    print('Current Run ID:', run_id)

    # Record args
    with open(log_dir+'/args.txt', 'a') as f:
        from pprint import pprint
        pprint(args.__dict__, f)

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

    train_model(
        dataset_label=args.dataset_label,
        task=args.task,
        seg_loss=args.segloss,
        model_label=args.model_label,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_dir=log_dir,
        experiment=experiment,
    )
