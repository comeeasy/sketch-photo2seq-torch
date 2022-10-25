"""
This script is copied from below github page

Thanks to 'grantbey!'
https://github.com/grantbey/PyTorch-SketchRNN/blob/master/sketch_rnn.py
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import numpy as np
from data.dataloader import *


class SeqEncoder(nn.Module):
    def __init__(self, config):
        super(SeqEncoder, self).__init__()
        
        self.encoder_hidden_size = config.hypers["encoder_hidden_size"]
        self.dropout = config.hypers["dropout"]
        self.Nz = config.hypers["Nz"]
        self.device = config.device
        
        # input_size = 5, there are 5 values for each point in the sequence
        self.lstm = nn.LSTM(input_size = 5, hidden_size = self.encoder_hidden_size, bias=True, dropout=self.dropout, bidirectional=True)
        self.mu = nn.Linear(2 * self.encoder_hidden_size, self.Nz)
        self.sigma = nn.Linear(2 * self.encoder_hidden_size, self.Nz)

    def forward(self,inputs,batch_size,hidden_cell=None):
        if hidden_cell is None:
            # Note: hidden state has dims num_layers * num_directions, batch, hidden_size
            # Here, there are 2 directions so thus a single layer
            # batch_size and hidden_size are already defined in params.
            hidden = Variable(torch.zeros(2, batch_size, self.encoder_hidden_size).to(self.device))
            cell =  Variable(torch.zeros(2, batch_size, self.encoder_hidden_size).to(self.device))
            hidden_cell = (hidden, cell)

        # Note: the input size is [131, 100, 5]
        # [Nmax, batch, seq_length]
        # or in torch notation: (seq_len, batch, input_size)

        (hidden,cell) = self.lstm(inputs.float(), hidden_cell)[1]
        # Split hidden in chunks of size = 1 along the first dimension
        # Since the first dimension is 2, it simply grabs each of these values
        # What's stopping using indexing? i.e. hidden_forward = hidden[0,...]
        # Then we don't need squeeze down below
        hidden_forward, hidden_reverse = torch.split(hidden,1,0)
        # size of hidden_forward / hidden_reverse will be [1,batch_size,encoder_hidden_size]
        # squeeze removes all dims of size 1, thus after squeeze they'll both be [batch_size,encoder_hidden_size]
        # concat on the second dimension, i.e. keep batches separate but concatenate hidden features
        hidden_cat = torch.cat([hidden_forward.squeeze(0),hidden_reverse.squeeze(0)],1)
        # Note that hidden_cat is [batch_size,2*encoder_hidden_size]
        mu = self.mu(hidden_cat)
        sigma = self.sigma(hidden_cat)
        # Additionally, z_size is also [batch_size,2*encoder_hidden_size]
        z_size = mu.size()

        # Make normal distributions, which are also [batch_size,2*encoder_hidden_size]
        N = Variable(torch.normal(torch.zeros(z_size), torch.ones(z_size)).to(self.device))

        # Combine mu, sigma and normal
        z = mu + N * torch.exp(sigma/2)
        # Note z has dimensions [batch_size,hyper.Nz] i.e. [100,128]

        return z, mu, sigma

class SeqDecoder(nn.Module):
    def __init__(self, config):
        super(SeqDecoder, self).__init__()
        
        self.Nz = config.hypers["Nz"]
        self.decoder_hidden_size = config.hypers["decoder_hidden_size"]
        self.dropout = config.hypers["dropout"]
        self.M = config.hypers["M"]
        
        # Once dataloader is decided, it needs Nmax of whole sketch data
        self.Nmax = None
        
        # A linear layer takes will take z and create the hidden/cell states
        # The input will be z, i.e. [batch_size,2*encoder_hidden_size] where 2*encoder_hidden_size = Nz
        # The output will be 2*params.decoder_hidden_size - this will be split into hidden/cell states below
        self.hc = nn.Linear(self.Nz, 2 * self.decoder_hidden_size)

        # Presumably the input_size = params.Nz+5 comes from the fact that the first point is added in,
        # thus the input size is 5 larger than the size of z
        self.lstm = nn.LSTM(input_size = self.Nz + 5, hidden_size = self.decoder_hidden_size, bias=True, dropout=self.dropout)

        # Another fully connected layer projects the hidden state of the LSTM to the output vector y
        # Unlike before, we won't use a non-linear activation here
        # The output is 5*M + M + 3
        # There are M bivariate normal distributions in the Gaussian mixture model that models (delta_x,delta_y)
        # Each bivariate normal distribution contains 5 parameters (hence 5*M)
        # There is another vector of length M which contains the mixture weights of the GMM
        # Finally, there is a categorical distribution (i.e. sums to 1) of size 3 that models the pen state (start line, end line, end drawing)
        # Thus, there are 6*M+3 parameters that need to be modelled for each line in a drawing.
        # Note that M is a hyperparameter.
        self.fc_y = nn.Linear(self.decoder_hidden_size, 6 * self.M + 3)

    def forward(self, inputs, z, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # Feed z into the linear layer, apply a tanh activation, then split along the second dimension
            # Since the size is 2*params.decoder_hidden_size, splitting is by params.decoder_hidden_size divides it into two parts
            hidden,cell = torch.split(F.tanh(self.hc(z)), self.decoder_hidden_size, 1)
            # Now create a tuple, add in an extra dimension in the first position and ensure that it's contiguous in memory
            hidden_cell = (hidden.unsqueeze(0).contiguous(),cell.unsqueeze(0).contiguous())

        # Note input size is [132, 100, 133]
        # This is [Nmax+1, batch, Nmax+1+1]
        # Where the Nmax+1+1 accounts for the fake initial value AND the concatenated z vector

        # Feed everything into the decoder LSTM
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        # Note: the output size will be [seq_len, batch, hidden_size * num_directions]
        # Thus, [132, batch, hyper.decoder_hidden_size] i.e. [132, 100, 512]

        # There are two modes: training and generate
        # While training, we feed the LSTM with the whole input and use all outputs
        # In generate mode, we just feed the last generated sample

        # Note: text implies that hidden state is used in training, whilst
        if self.training:
            # Note: view(-1,...) reshapes the output to a vector of length params.dec_hidden_size
            # i.e. [132, 100, 512] -> [13200, 512]
            y = self.fc_y(outputs.view(-1, self.decoder_hidden_size))
        else:
            y = self.fc_y(hidden.view(-1, self.decoder_hidden_size))

        # Note y has size [batch*(Nmax+1),hyper.decoder_hidden_size] i.e [13200, 512]
        # Split the output into groups of 6 parameters along the second axis
        # Then stack all but the last one
        # This creates params_mixture of size [M, (Nmax+1)*batch, 6], i.e. the 5 parameters of the bivariate normal distribution and the mixture weight
        # for each of the Nmax lines in each of the batches
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1])

        # Finally, the last three values are the parameters of the pen at this particular point
        # This has a size [(Nmax+1)*batch, 3]
        params_pen = params[-1]

        # Now split each parameter and label the variables appropriately
        # Each will be of size [M, (Nmax+1)*batch, 1]

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        # Note: Nmax = 131
        if self.training:
            len_out = self.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,self.M)
        q = F.softmax(params_pen).view(len_out,-1,3)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

"""
    References: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""


"""
    Template of Pix Encoder and Pix Decoder
    
    each backbone model must be declared in other class form.
"""
 
class PixEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=None, Nz=128) -> None:
        super(PixEncoder, self).__init__()
        
        self.latent_dim = Nz
        self.hidden_dims = hidden_dims
        
        if hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            in_channels = h_dim
        modules.append(
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h.view(-1, self.hidden_dims[-1])), self.fc_var(h.view(-1, self.hidden_dims[-1]))
        std = torch.exp(0.5 * logvar)        
        z = self.reparameterize(mu, std)
        return z, mu, std
           

class PixDecoder(nn.Module):
    def __init__(self, hidden_dims=None, Nz=128, out_channels=3) -> None:
        super(PixDecoder, self).__init__()
        
        self.latent_dim = Nz
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        
        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * 7 * 7)
        
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(inplace=True)    
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], hidden_dims[-1],
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

        
    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 512, 7, 7)
        out = self.decoder(out)
        out = self.final_layer(out)
        
        return out

class Model():
    def __init__(self, config):
        
        self.lr         = config.hypers["lr"]
        self.eta_min    = config.hypers["eta_min"]
        self.eta_step   = self.eta_min
        self.batch_size = config.hypers["batch_size"]
        self.R          = config.hypers["R"]
        self.grad_clip  = config.hypers["grad_clip"]
        self.min_lr     = config.hypers["min_lr"]
        self._lr_decay  = config.hypers["lr_decay"]
        self.KL_min     = config.hypers["KL_min"]
        self.wKL        = config.hypers["wKL"]
        self.device     = config.device
        self.Nz         = config.hypers["Nz"]
        self.quick_draw_save = config.train["quick_draw"]["weights_save"]
        self.qmul_save  = config.train["qmul"]["weights_save"]
        self.quick_draw_resume = config.train["quick_draw"]["resume"]
        self.qmul_resume = config.train["qmul"]["resume"]
        
        self.seq_enc = SeqEncoder(config).to(self.device)
        self.seq_dec = SeqDecoder(config).to(self.device)
        self.pix_enc = PixEncoder().to(self.device)
        self.pix_dec = PixDecoder().to(self.device)
        
        if self.quick_draw_resume:
            start_epoch = config.train["quick_draw"]["start_epoch"]
            seq_enc_path = os.path.join(self.quick_draw_save, f"seq_enc_{start_epoch}.pt")
            seq_dec_path = os.path.join(self.quick_draw_save, f"seq_dec_{start_epoch}.pt")
            self.seq_enc.load_state_dict(torch.load(seq_enc_path))
            self.seq_dec.load_state_dict(torch.load(seq_dec_path))
        if self.qmul_resume:
            start_epoch = config.train["qmul"]["start_epoch"]
            seq_enc_path = os.path.join(self.quick_draw_save, f"seq_enc_{start_epoch}.pt")
            seq_dec_path = os.path.join(self.quick_draw_save, f"seq_dec_{start_epoch}.pt")
            pix_enc_path = os.path.join(self.quick_draw_save, f"pix_enc_{start_epoch}.pt")
            pix_dec_path = os.path.join(self.quick_draw_save, f"pix_dec_{start_epoch}.pt")
            self.seq_enc.load_state_dict(torch.load(seq_enc_path))
            self.seq_dec.load_state_dict(torch.load(seq_dec_path))
            self.pix_enc.load_state_dict(torch.load(pix_enc_path))
            self.pix_dec.load_state_dict(torch.load(pix_dec_path))
        
        self.seq_enc_optim = optim.Adam(self.seq_enc.parameters(),self.lr)
        self.seq_dec_optim = optim.Adam(self.seq_dec.parameters(),self.lr)
        self.pix_enc_optim = optim.Adam(self.pix_enc.parameters(), self.lr)
        self.pix_dec_optim = optim.Adam(self.pix_dec.parameters(), self.lr)
        

    def quick_draw_train(self, dataloader, epoch, writer):
        self.Nmax = dataloader.Nmax
        self.seq_dec.Nmax = self.Nmax
        
        self.seq_enc.train()
        self.seq_dec.train()

        batch, lengths = dataloader.get_batch(self.batch_size)

        z, self.mu, self.sigma = self.seq_enc(batch, self.batch_size)

        sos = Variable(torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.batch_size).cuda()).unsqueeze(0)
        batch_init = torch.cat([sos, batch], 0)
        z_stack = torch.stack([z] * (self.Nmax + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.seq_dec(inputs, z, self.batch_size)

        mask, dx, dy, p = dataloader.get_target(batch,lengths)

        self.seq_enc_optim.zero_grad()
        self.seq_dec_optim.zero_grad()

        self.eta_step = 1 - (1 - self.eta_min) * self.R

        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p)
        loss = LR + LKL

        writer.add_scalar("QuickDraw/Total/Loss", loss, epoch)
        writer.add_scalar("QuickDraw/KL/s2s", LKL, epoch)
        writer.add_scalar("QuickDraw/Reconstruction/s2s", LR, epoch)

        loss.backward()

        nn.utils.clip_grad_norm(self.seq_enc.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm(self.seq_dec.parameters(), self.grad_clip)

        self.seq_enc_optim.step()
        self.seq_dec_optim.step()

        if epoch%500 == 0:
            self.seq_enc_optim = self.lr_decay(self.seq_enc_optim)
            self.seq_dec_optim = self.lr_decay(self.seq_dec_optim)

        if epoch % 10000 == 0:
            pt_name = f"qd_seq_dec_{epoch}.pt"
            torch.save(self.seq_dec.state_dict(), os.path.join(self.quick_draw_save, pt_name))

        # todo save

        # todo load

        # todo conditional generation
        # This uses z from a trained model encoder, but feeds a sample image into the decoder
        # Using the input, the model "reconstructs" the image
        # Thus, not deterministic, but random
        # Temperature parameter controls randomness

        # todo unconditional generation
        # hidden/cell are initialized to zero and no z vector is used
        # encoder is _not_ trained
        # Can sample images and vary temp to get more varied output
        

    def qmul_train(self, dataloader, epoch, writer):
        self.Nmax = dataloader.Nmax
        self.seq_dec.Nmax = self.Nmax
        
        self.seq_enc.train()
        self.seq_dec.train()
        self.pix_enc.train()
        self.pix_dec.train()

        batch, lengths, batch_images = dataloader.get_batch(self.batch_size)
        
        # seq2seq
        z, self.mu, self.sigma = self.seq_enc(batch, self.batch_size)
        sos = Variable(torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.batch_size).to(self.device)).unsqueeze(0)
        batch_init = torch.cat([sos, batch], 0)
        z_stack = torch.stack([z] * (self.Nmax + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.seq_dec(inputs, z, self.batch_size)
        mask, dx, dy, p = dataloader.get_target(batch,lengths)
        
        L_s2s_KL = self.kullback_leibler_loss()
        L_s2s_R = self.reconstruction_loss(mask, dx, dy, p)

        # pix2pix
        z, self.mu, self.sigma = self.pix_enc(batch_images)
        pred_imgs = self.pix_dec(z)
        
        L_p2p_KL = self.kullback_leibler_loss()
        L_p2p_R = F.mse_loss(pred_imgs, batch_images)
        
        # seq2pix
        z, self.mu, self.sigma = self.seq_enc(batch, self.batch_size)
        pred_imgs = self.pix_dec(z)
        
        L_s2p_KL = self.kullback_leibler_loss()
        L_s2p_R = F.mse_loss(pred_imgs, batch_images)
        
        # pix2seq
        z, self.mu, self.sigma = self.pix_enc(batch_images)
        
        sos = Variable(torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.batch_size).to(self.device)).unsqueeze(0)
        batch_init = torch.cat([sos, batch], 0)
        z_stack = torch.stack([z] * (self.Nmax + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.seq_dec(inputs, z, self.batch_size)
        mask, dx, dy, p = dataloader.get_target(batch,lengths)
        
        L_p2s_KL = self.kullback_leibler_loss()
        L_p2s_R = self.reconstruction_loss(mask, dx, dy, p)

        self.seq_enc_optim.zero_grad()
        self.seq_dec_optim.zero_grad()
        self.pix_enc_optim.zero_grad()
        self.pix_dec_optim.zero_grad()

        self.eta_step = 1 - (1 - self.eta_min) * self.R

        L_KL = L_s2s_KL + L_p2p_KL + L_s2p_KL + L_p2s_KL
        L_shortcut = L_s2s_R + L_p2p_R
        L_R = L_s2p_R + L_p2s_R
        loss = L_R + L_shortcut + L_KL

        writer.add_scalar("QMUL/Total/Loss", loss, epoch)
        writer.add_scalar("QMUL/KL/s2s", L_s2s_KL, epoch)
        writer.add_scalar("QMUL/KL/p2p", L_p2p_KL, epoch)
        writer.add_scalar("QMUL/KL/s2p", L_s2p_KL, epoch)
        writer.add_scalar("QMUL/KL/p2s", L_p2s_KL, epoch)
        writer.add_scalar("QMUL/Reconstruction/s2s", L_s2s_R, epoch)
        writer.add_scalar("QMUL/Reconstruction/p2p", L_p2p_R, epoch)
        writer.add_scalar("QMUL/Reconstruction/s2p", L_s2p_R, epoch)
        writer.add_scalar("QMUL/Reconstruction/p2s", L_p2s_R, epoch)
        
        loss.backward()

        nn.utils.clip_grad_norm_(self.seq_enc.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.seq_dec.parameters(), self.grad_clip)

        self.seq_enc_optim.step()
        self.seq_dec_optim.step()
        self.pix_enc_optim.step()
        self.pix_dec_optim.step()

        if epoch%500 == 0:
            self.seq_enc_optim = self.lr_decay(self.seq_enc_optim)
            self.seq_dec_optim = self.lr_decay(self.seq_dec_optim)
            self.pix_enc_optim = self.lr_decay(self.pix_enc_optim)
            self.pix_dec_optim = self.lr_decay(self.pix_dec_optim)
        
            pix_enc_name = f"pic_enc_{epoch}.pt"
            seq_dec_name = f"seq_dec_{epoch}.pt"
            torch.save(self.pix_enc.state_dict(), os.path.join(self.qmul_save, pix_enc_name))
            torch.save(self.seq_dec.state_dict(), os.path.join(self.qmul_save, seq_dec_name))

        # todo conditional generation
        # This uses z from a trained model encoder, but feeds a sample image into the decoder
        # Using the input, the model "reconstructs" the image
        # Thus, not deterministic, but random
        # Temperature parameter controls randomness

        # todo unconditional generation
        # hidden/cell are initialized to zero and no z vector is used
        # encoder is _not_ trained
        # Can sample images and vary temp to get more varied output


    def lr_decay(self, optimizer):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > self.min_lr:
                param_group['lr'] *= self._lr_decay
        return optimizer

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy ** 2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, p):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * pdf, 2))) / float(self.Nmax * self.batch_size)
        LP = -torch.sum(p * torch.log(self.q)) / float(self.Nmax * self.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma))/ float(self.Nz * self.batch_size)
        KL_min = Variable(torch.Tensor([self.KL_min]).to(self.device)).detach()
        return self.wKL * self.eta_step * torch.max(LKL, KL_min)