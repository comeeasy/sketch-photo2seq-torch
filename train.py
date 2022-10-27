import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os, time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.model import Model
from config import Config
from data.dataloader import QMULLoader, QuickDrawLoader


def get_args():
    parser = argparse.ArgumentParser(description="e.g.) python train.py /path/to/config.yml")
    parser.add_argument("--config", required=True, help="config.yml path")
    
    return parser.parse_args()


def main():
    args = get_args()
    config = Config(args.config)
    
    QuickDraw_epochs = config.train["quick_draw"]["epochs"]
    QMUL_epochs      = config.train["qmul"]["epochs"]
    tensorboard_log  = config.train["tensorboard_log"]
    tensorboard_log  = os.path.join(tensorboard_log, time.strftime('%Y%m%H%M', time.localtime()))

    model = Model(config)
    quick_draw_loader = QuickDrawLoader(config)
    qmul_loader       = QMULLoader(config)
    
    with SummaryWriter(tensorboard_log) as writer:
        # pre-training
        if config.train["quick_draw"]["resume"]: start_epoch = config.train["quick_draw"]["start_epoch"]
        else: start_epoch = 1   
        for epoch in tqdm(range(start_epoch, QuickDraw_epochs + start_epoch)):
            model.quick_draw_train(quick_draw_loader, epoch, writer)
        
        # train
        if config.train["qmul"]["resume"]: start_epoch = config.train["qmul"]["start_epoch"]
        else: start_epoch = 1   
        for epoch in tqdm(range(start_epoch, QMUL_epochs + start_epoch)):
            model.qmul_train(qmul_loader, epoch, writer)
    

if __name__=="__main__":
    main()