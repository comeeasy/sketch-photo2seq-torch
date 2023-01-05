import glob
import logging
import os
import re

import h5py
import numpy as np
import torch
# import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from svgpathtools import svg2paths
from torch.autograd import Variable
from torch.utils.data import Dataset


class SketchDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.max_seq_length = config["hypers"]["max_seq_length"]
        self.M = config["hypers"]["M"]
        self.device = config["device"]
    
    def purify(self, strokes):
        data = []
        for seq in strokes:
            if len(seq[:, 0]) <= self.max_seq_length and len(seq[:, 0]) > 10:
                seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype = np.float32)
            data.append(seq)
        return data
    def calculate_normalizing_scale_factor(self, strokes):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(strokes)):
            for j in range(len(strokes[i])):
                data.append(strokes[i][j, 0])
                data.append(strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)
    def normalize(self, strokes):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:2] /= scale_factor
            data.append(seq)
        return data
    def max_size(self, strokes):
        """larger sequence length in the data set"""
        sizes = [len(seq) for seq in strokes]
        return max(sizes)


class QuickDrawDataset(Dataset):
    def __init__(self, config):
        """
            args:
                npz_path: path of npz_path. e.g.) "/home/joono/MinLab/sketch-photo2seq-torch/datasets/QuickDraw/shoes/npz/shoe.npz"
                train_valid_test: "train" or "valid" or "test"
        """
        super().__init__()
        
        self.max_seq_length = config["hypers"]["max_seq_length"]
        self.M = config["hypers"]["M"]
        self.datafile = config["data"]["quick_draw"]
        self.device = config["device"]
        
        def purify(self, strokes):
            data = []
            for seq in strokes:
                if len(seq[:, 0]) <= self.max_seq_length and len(seq[:, 0]) > 10:
                    seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype = np.float32)
                data.append(seq)
            return data
        def calculate_normalizing_scale_factor(self, strokes):
            """Calculate the normalizing factor explained in appendix of sketch-rnn."""
            data = []
            for i in range(len(strokes)):
                for j in range(len(strokes[i])):
                    data.append(strokes[i][j, 0])
                    data.append(strokes[i][j, 1])
            data = np.array(data)
            return np.std(data)
        def normalize(self, strokes):
            """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
            data = []
            scale_factor = calculate_normalizing_scale_factor(self,strokes)
            for seq in strokes:
                seq[:, 0:2] /= scale_factor
                data.append(seq)
            return data
        def max_size(self, strokes):
            """larger sequence length in the data set"""
            sizes = [len(seq) for seq in strokes]
            return max(sizes)
        
        self.data = np.load(self.datafile, encoding = 'latin1', allow_pickle=True) 
        self.data = self.data['train']
        self.data = purify(self, self.data)
        self.data = normalize(self, self.data)
        self.Nmax = max_size(self, self.data)
        print(f"Nmax: {self.Nmax}")
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        stroke = self.data[index]
                
        len_seq = len(stroke)
        new_seq = np.zeros((self.Nmax, 5))
        new_seq[:len_seq, :2] = stroke[:, :2]  # fill in x:y co-ords in first two dims
        new_seq[:len_seq - 1, 2] = 1 - stroke[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
        new_seq[:len_seq, 3] = stroke[:, 2]  # pen binary in fourth dim
        new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
        new_seq[len_seq - 1, 2:4] = 0
        
        stroke = torch.from_numpy(new_seq).float()
        length = len_seq
    
        eos = Variable(torch.stack([torch.Tensor([0, 0, 0, 0, 1])]))
        stroke_target = torch.cat([stroke, eos], 0)
        mask = torch.zeros(self.Nmax + 1)
        
        mask[:length] = 1
        mask = Variable(mask).detach()
    
        dx = torch.stack([Variable(stroke_target[:, 0])] * self.M, 1).detach()
        dy = torch.stack([Variable(stroke_target[:, 1])] * self.M, 1).detach()
        p1 = Variable(stroke_target[:, 2]).detach()
        p2 = Variable(stroke_target[:, 3]).detach()
        p3 = Variable(stroke_target[:, 4]).detach()
        p = torch.stack([p1, p2, p3], 1)
    
        return stroke, length, (mask, dx, dy, p) 
    
class QMULDataset(Dataset):
    """
        args:
            h5_path: path of h5_path. e.g.) "/home/joono/MinLab/sketch-photo2seq-torch/datasets/QMUL/shoe"
            train_valid_test: "train" or "valid" or "test"
    """
    def __init__(self, config, train=True):
        super().__init__()
        
        self.max_seq_length = config["hypers"]["max_seq_length"]
        
        if train: self.datafile = config.data["qmul_train"]
        else    : self.datafile = config.data["qmul_test"]
            
        self.M = config["hypers"]["M"]
        self.device = config["device"]
        self.qmul_image_path = os.path.join("datasets", "QMUL", "shoes", "photos")
        self.img_size = config["hypers"]["img_size"]
        self.img_crop = config["hypers"]["img_crop"]

        def purify(self, strokes):
            data = []
            for seq in strokes:
                if len(seq[:, 0]) <= self.max_seq_length and len(seq[:, 0]) > 10:
                    seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype = np.float32)
                data.append(seq)
            return data
        def calculate_normalizing_scale_factor(self, strokes):
            """Calculate the normalizing factor explained in appendix of sketch-rnn."""
            data = []
            for i in range(len(strokes)):
                for j in range(len(strokes[i])):
                    data.append(strokes[i][j, 0])
                    data.append(strokes[i][j, 1])
            data = np.array(data)
            return np.std(data)
        def normalize(self, strokes):
            """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
            data = []
            scale_factor = calculate_normalizing_scale_factor(self,strokes)
            for seq in strokes:
                seq[:, 0:2] /= scale_factor
                data.append(seq)
            return data
        def max_size(self, strokes):
            """larger sequence length in the data set"""
            sizes = [len(seq) for seq in strokes]
            return max(sizes)
        
        def load_h5(datafile):
            with h5py.File(datafile, 'r') as hf:
                d = {key: np.array(hf.get(key)) for key in hf.keys()}
            d["data_offset"].shape, d["image_data"].shape
            data = []
            for off_s, off_e in d["data_offset"]:
                data.append(d["image_data"][off_s:off_e, :-1])
            return np.array(data), d["image_base_name"]
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop((self.img_crop, self.img_crop)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.data, self.image_base_name = load_h5(self.datafile)    
        self.data = purify(self,self.data)
        self.data = normalize(self,self.data)
        self.Nmax = max_size(self,self.data)
    
    # https://github.com/MarkMoHR/sketch-photo2seq/blob/master/utils.py
    def _load_data(self, fname):
        with h5py.File(fname, 'r') as hf:
            d = {key: np.array(hf.get(key)) for key in hf.keys()}
    
        num_data = len(d["attribute"])
        strokes = np.zeros((num_data, self.max_strokes, 5))
        strokes[:, :, -1] = 1.
        for i in range(num_data):
            _s = d["image_data"][d["data_offset"][i][0]:d["data_offset"][i][1]]
            len_s = len(_s)
            _offset, _p = _s[:, :2], _s[:, -2]
            strokes[i, :len_s, :2] = _offset
            
            for j in range(len_s):
                strokes[i, j, 2:] = np.eye(3)[_p[j].astype(np.int8)]
            
        return d['png_data'].astype(np.float32), strokes.astype(np.float32)
        
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        image_base_name = self.image_base_name[index]
        image_name = image_base_name[0].decode("utf-8") + ".png"
        img_path = os.path.join(self.qmul_image_path, image_name)
        pil_img = Image.open(img_path)
        tensor_img = self.transforms(pil_img)
        
        stroke = self.data[index]
                
        len_seq = len(stroke)
        new_seq = np.zeros((self.Nmax, 5))
        new_seq[:len_seq, :2] = stroke[:, :2]  # fill in x:y co-ords in first two dims
        new_seq[:len_seq - 1, 2] = 1 - stroke[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
        new_seq[:len_seq, 3] = stroke[:, 2]  # pen binary in fourth dim
        new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
        new_seq[len_seq - 1, 2:4] = 0
        
        stroke = torch.from_numpy(new_seq).float()
        length = len_seq
    
        eos = Variable(torch.stack([torch.Tensor([0, 0, 0, 0, 1])]))
        stroke_target = torch.cat([stroke, eos], 0)
        mask = torch.zeros(self.Nmax + 1)
        
        mask[:length] = 1
        mask = Variable(mask).detach()
    
        dx = torch.stack([Variable(stroke_target[:, 0])] * self.M, 1).detach()
        dy = torch.stack([Variable(stroke_target[:, 1])] * self.M, 1).detach()
        p1 = Variable(stroke_target[:, 2]).detach()
        p2 = Variable(stroke_target[:, 3]).detach()
        p3 = Variable(stroke_target[:, 4]).detach()
        p = torch.stack([p1, p2, p3], 1)
        
        return stroke, length, tensor_img, (mask, dx, dy, p) 
    
    
class PortraitDataset(SketchDataset):
    """
        It returns as below
        
    """
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        # np.array data
        fname = config["data"]["portrait_npy"]
        print(f"load npy file in {fname}")
        self.data = self._load_npy(fname)
        self.data = self.normalize(self.data)
        self.Nmax = self.max_size(self.data)
        self.images = self._load_pil(config["data"]["portrait_png"])

        # each number of datasets are must be same
        print(f"svg: {len(self.data)} png: {len(self.images)}")
        assert len(self.data) == len(self.images)

        # transforms
        self.img_size = config["hypers"]["img_size"]
        self.img_crop = config["hypers"]["img_crop"]
        mean, std = [[0.5] for _ in range(config["hypers"]["in_channels"])], [[0.5] for _ in range(config["hypers"]["in_channels"])]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop((self.img_crop, self.img_crop)),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = self.transforms(img)
        
        stroke = self.data[index]
                
        len_seq = len(stroke)
        new_seq = np.zeros((self.Nmax, 5))
        new_seq[:len_seq, :2] = stroke[:, :2]  # fill in x:y co-ords in first two dims
        new_seq[:len_seq - 1, 2] = 1 - stroke[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
        new_seq[:len_seq, 3] = stroke[:, 2]  # pen binary in fourth dim
        new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
        new_seq[len_seq - 1, 2:4] = 0
        
        stroke = torch.from_numpy(new_seq).float()
        length = len_seq
    
        eos = Variable(torch.stack([torch.Tensor([0, 0, 0, 0, 1])]))
        stroke_target = torch.cat([stroke, eos], 0)
        mask = torch.zeros(self.Nmax + 1)
        
        mask[:length] = 1
        mask = Variable(mask).detach()
    
        dx = torch.stack([Variable(stroke_target[:, 0])] * self.M, 1).detach()
        dy = torch.stack([Variable(stroke_target[:, 1])] * self.M, 1).detach()
        p1 = Variable(stroke_target[:, 2]).detach()
        p2 = Variable(stroke_target[:, 3]).detach()
        p3 = Variable(stroke_target[:, 4]).detach()
        p = torch.stack([p1, p2, p3], 1)
        
        return stroke, length, img, (mask, dx, dy, p) 
        
    
    def _load_pil(self, png_path):
        png_files = sorted(glob.glob(os.path.join(png_path, "*.png")), key=lambda x: list(map(int, re.findall(r"\d+", x)))[0])
        
        pil_imgs = []
        for png_file in png_files:
            if self.config["hypers"]["in_channels"] == 1:
                pil_imgs.append(Image.open(png_file).convert('L'))
            elif self.config["hypers"]["in_channels"] == 3:
                pil_imgs.append(Image.open(png_file).convert('RGB'))
        return pil_imgs
    
    def _load_npy(self, fname):
        data = np.load(fname, allow_pickle=True)
        return np.array([d["strokes"][0] for d in data[:, 1]])
            
    def _svg_dir2npy_data(self, svg_path) -> np.array:
        '''
            svg files are preprocessed with code that https://github.com/UmarSpa/PNG-to-SVG
            and bèzier curve to line segments (custom code)
        '''
        
        p = re.compile(r"[ML]+ \d*[.]*\d*,\d*[.]*\d*")
        
        svg_files = sorted(glob.glob(os.path.join(svg_path, "*.svg")), key=lambda x: list(map(int, re.findall(r"\d+", x)))[0])
        
        print(len(svg_files))
        print(svg_files)
        
        dset = []
        for line_svg in svg_files:
            _, attributes = svg2paths(line_svg)
            abs_dset = []
            for att in attributes:
                attr = att['d']
                pen_movements = p.findall(attr)
                for pen_move in pen_movements:
                    pen_s = pen_move.split(" ")[0]
                    if pen_s == "M": pen_state = 1.
                    elif pen_s == "L": pen_state = 0.
                    else: raise RuntimeError(f"pen_s: {pen_s}")
                    abs_x, abs_y = map(float, pen_move.split(" ")[1].split(","))
                    abs_dset.append(np.array((abs_x, abs_y, pen_state)))

            strokes = []
            strokes.append(abs_dset[0])
            for i in range(1, len(abs_dset)):
                delta_x = abs_dset[i][0] - abs_dset[i-1][0]
                delta_y = abs_dset[i][1] - abs_dset[i-1][1]
                pen_state = abs_dset[i][2]
                
                strokes.append((delta_x, delta_y, pen_state))
            
            dset.append(np.array(strokes))

        return dset
            
            
            