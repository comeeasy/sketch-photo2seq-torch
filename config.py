from ast import Load
import torch
import yaml

class Config():
    def __init__(self, config_path=None):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.hypers = config["hypers"]
        self.data = config["data"]
        self.train = config["train"]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
