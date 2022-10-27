import os
import numpy as np
import torch
import torchvision.transforms as transforms
import h5py

from torch.autograd import Variable
from PIL import Image

class QuickDrawLoader():
    def __init__(self, config):

        self.max_seq_length = config.hypers["max_seq_length"]
        self.M = config.hypers["M"]
        self.datafile = config.data["quick_draw"]
        self.device = config.device

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
        # self.data = purify(self, self.data)
        # self.data = normalize(self, self.data)
        # self.Nmax = max_size(self, self.data)
        self.data = purify(self, self.data)
        self.data = normalize(self, self.data)
        self.Nmax = max_size(self, self.data)

    def get_batch(self, batch_size):
        idxs = np.random.choice(len(self.data),batch_size)
        batch_strokes = [self.data[idx] for idx in idxs]
        strokes = []
        lengths = []
        for seq in batch_strokes:
            len_seq = len(seq[:, 0])  # I think this is how many lines in the image
            # Seq is always of shape (n,3) where the three dimensions
            # ∆x, ∆y, and a binary value representing whether the pen is lifted away from the paper
            new_seq = np.zeros((self.Nmax, 5))  # New seq of max length, all zeros
            new_seq[:len_seq, :2] = seq[:, :2]  # fill in x:y co-ords in first two dims
            new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
            new_seq[:len_seq, 3] = seq[:, 2]  # pen binary in fourth dim
            new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
            new_seq[len_seq - 1, 2:4] = 0  # zeros in last point for dims three and four
            lengths.append(len(seq[:, 0]))  # Record the length of the actual sequence
            strokes.append(new_seq)  # Record the sequence

        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).to(self.device).float())

        return batch, lengths

    def get_target(self, batch, lengths):
        eos = Variable(torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).to(self.device)).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.Nmax + 1, batch.size()[1])

        for id, length in enumerate(lengths):
            mask[:length,id] = 1
        mask = Variable(mask.to(self.device)).detach()

        dx = torch.stack([Variable(batch.data[:, :, 0])] * self.M, 2).detach()
        dy = torch.stack([Variable(batch.data[:, :, 1])] * self.M, 2).detach()
        p1 = Variable(batch.data[:, :, 2]).detach()
        p2 = Variable(batch.data[:, :, 3]).detach()
        p3 = Variable(batch.data[:, :, 4]).detach()
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p
    
class QMULLoader():
    def __init__(self, config, train=True):

        self.max_seq_length = config.hypers["max_seq_length"]
        
        if train: self.datafile = config.data["qmul_train"]
        else    : self.datafile = config.data["qmul_test"]
            
        self.M = config.hypers["M"]
        self.device = config.device
        self.qmul_image_path = os.path.join("datasets", "QMUL", "shoes", "photos")

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
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.data, self.image_base_name = load_h5(self.datafile)    
        self.data = purify(self,self.data)
        self.data = normalize(self,self.data)
        self.Nmax = max_size(self,self.data)

    def get_batch(self, batch_size):
        idxs = np.random.choice(len(self.data),batch_size)
        batch_strokes = [self.data[idx] for idx in idxs]
        
        batch_image_base_name = [self.image_base_name[idx] for idx in idxs]
        batch_images = []
        for image_base_name in batch_image_base_name:
            image_name = image_base_name[0].decode("utf-8") + ".png"
            img_path = os.path.join(self.qmul_image_path, image_name)
            pil_img = Image.open(img_path)
            tensor_img = self.transforms(pil_img)
            batch_images.append(tensor_img)
        batch_images = torch.stack(batch_images, dim=0).to(self.device)
        
        strokes = []
        lengths = []
        for seq in batch_strokes:
            len_seq = len(seq[:, 0])  # I think this is how many lines in the image
            # Seq is always of shape (n,3) where the three dimensions
            # ∆x, ∆y, and a binary value representing whether the pen is lifted away from the paper
            new_seq = np.zeros((self.Nmax, 5))  # New seq of max length, all zeros
            new_seq[:len_seq, :2] = seq[:, :2]  # fill in x:y co-ords in first two dims
            new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
            new_seq[:len_seq, 3] = seq[:, 2]  # pen binary in fourth dim
            new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
            new_seq[len_seq - 1, 2:4] = 0  # zeros in last point for dims three and four
            lengths.append(len(seq[:, 0]))  # Record the length of the actual sequence
            strokes.append(new_seq)  # Record the sequence

        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).to(self.device).float())

        return batch, lengths, batch_images

    def get_target(self, batch, lengths):
        eos = Variable(torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).to(self.device)).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.Nmax + 1, batch.size()[1])

        for id, length in enumerate(lengths):
            mask[:length,id] = 1
        mask = Variable(mask.to(self.device)).detach()

        dx = torch.stack([Variable(batch.data[:, :, 0])] * self.M, 2).detach()
        dy = torch.stack([Variable(batch.data[:, :, 1])] * self.M, 2).detach()
        p1 = Variable(batch.data[:, :, 2]).detach()
        p2 = Variable(batch.data[:, :, 3]).detach()
        p3 = Variable(batch.data[:, :, 4]).detach()
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p
    
    
