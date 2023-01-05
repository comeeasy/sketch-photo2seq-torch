import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import os, time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.model import Model
from config import Config
from data.custom_dataloader import QuickDrawDataset, QMULDataset, PortraitDataset



def get_args():
    parser = argparse.ArgumentParser(description="e.g.) python train.py /path/to/config.yml")
    parser.add_argument("--config", required=True, help="config.yml path")
    
    return parser.parse_args()

def load_weights(attr, model):
    seq_enc_path = os.path.join(attr["weights_save"], f"seq_enc_{attr['start_epoch']}.pt")
    seq_dec_path = os.path.join(attr["weights_save"], f"seq_dec_{attr['start_epoch']}.pt")
    model.seq_enc.load_state_dict(torch.load(seq_enc_path))
    model.seq_dec.load_state_dict(torch.load(seq_dec_path))
        

def get_dataloader(config, dataset_name):
    if dataset_name == "qmul":
        dset = QMULDataset(config)
        Nmax = dset.Nmax
        loader = DataLoader(
            dataset=dset,
            batch_size=config["hypers"]["batch_size"],
            shuffle=True,
            num_workers=1,
            drop_last=True,
            pin_memory=True
        )
    elif dataset_name == "portrait":
        dset = PortraitDataset(config)
        Nmax = dset.Nmax
        loader = DataLoader(
            dataset=dset,
            batch_size=config["hypers"]["batch_size"],
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    elif dataset_name == "quickdraw":
        dset = QuickDrawDataset(config)
        Nmax = dset.Nmax
        loader = DataLoader(
            dataset=dset,
            batch_size=config["hypers"]["batch_size"],
            shuffle=True,
            num_workers=16,
            drop_last=True,
        )
    
    return loader, Nmax
    

def main():
    args = get_args()
    config = Config(args.config).get_config()
    tensorboard_log  = os.path.join(config["tensorboard_log"], time.strftime('%D/%M', time.localtime()))

    torch.autograd.set_detect_anomaly(True)

    model = Model(config)
    with SummaryWriter(tensorboard_log) as writer:
        for dataset_name, attr in config["train"].items():
            if not os.path.isdir(attr["weights_save"]):
                os.mkdir(attr["weights_save"])
            
            start_epoch = 1
            if attr["resume"]:
                start_epoch = attr["start_epoch"]
                load_weights(attr=attr, model=model)
                
            loader, Nmax = get_dataloader(config=config, dataset_name=dataset_name)
            for epoch in tqdm(range(start_epoch, attr["epochs"] + start_epoch)):
                model.s2s_train(attr, loader, epoch, Nmax, writer)
                    

if __name__=="__main__":
    main()