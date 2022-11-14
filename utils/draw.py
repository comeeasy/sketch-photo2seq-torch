# import the required libraries
import numpy as np
import time
import random
# import cPickle
import codecs
import collections
import os
import math
import json
# import tensorflow as tf
# from six.moves import xrange

import svgwrite

from IPython.display import SVG, display
import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

# https://github.com/magenta/magenta/blob/main/magenta/models/sketch_rnn/utils.py
def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename='./tmp/sketch_rnn/svg/sample.svg', draw=True):
    # tf.gfile.MakeDirs(os.path.dirname(svg_filename))
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x 
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        lift_pen = data[i, 2] 
        
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0])/factor
        y = float(data[i,1])/factor
        p += command+str(x)+","+str(y)+" "
    the_color = "black"
    stroke_width = 1
    
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()
    if draw:
        display(SVG(dwg.tostring()))

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max+x_min)*0.5
        return x_start-center_loc, x_end
    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0]*grid_space+grid_space*0.5
        grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x+loc_x
        new_y_pos = grid_y+loc_y
        result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos+delta_pos[0]
        y_pos = new_y_pos+delta_pos[1]
    return np.array(result)

# https://github.com/MarkMoHR/sketch-photo2seq/blob/master/sketch_p2s_sampling.py
def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i][4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result

def sample(pix_enc, seq_dec, images, device, max_len=250, temperature=1.0, greedy_mode=False):
    """Samples a sequence from a pre-trained model."""

    def adjust_temp(pi_pdf, temp):
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """Samples from a pdf, optionally greedily."""
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_temp(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        print('Error with sampling ensemble.')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    z, _, _ = pix_enc(images)

    s = Variable(torch.stack([torch.Tensor([0, 0, 1, 0, 0])]).to(device)).unsqueeze(0)
    batch_init = torch.cat([s])
    z_stack = torch.stack([z])
    inputs = torch.cat([batch_init, z_stack], 2)

    o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, hidden, cell = seq_dec(inputs, z, 1) # batch size is 1
    pen_state = torch.argmax(o_pen)

    greedy = greedy_mode
    temp = temperature

    strokes = []
    iter = 0
    while pen_state != 2 and iter < max_len:
        iter += 1
        o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, hidden, cell = seq_dec(inputs, z, 1, (hidden, cell))

        # [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = gmm_coef
        # top 6 param: [1, 20], o_pen: [1, 3], next_state: [1, 1024]

        o_pi = o_pi.squeeze(0).detach().cpu().numpy()
        o_mu1 = o_mu1.squeeze(0).detach().cpu().numpy()
        o_mu2 = o_mu2.squeeze(0).detach().cpu().numpy()
        o_sigma1 = o_sigma1.squeeze(0).detach().cpu().numpy()
        o_sigma2 = o_sigma2.squeeze(0).detach().cpu().numpy()
        o_corr = o_corr.squeeze(0).detach().cpu().numpy()
        o_pen = o_pen.squeeze(0).detach().cpu().numpy()

        idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)
        pen_state = get_pi_idx(random.random(), o_pen[0], temp, greedy)

        eos = [0, 0, 0]
        eos[pen_state] = 1
        next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                              o_sigma1[0][idx], o_sigma2[0][idx],
                                              o_corr[0][idx], np.sqrt(temp), greedy)

        strokes.append((next_x1, next_x2, eos[0], eos[1], eos[2]))
        s[0, 0, 0] = next_x1
        s[0, 0, 1] = next_x2
        s[0, 0, 2:] = torch.eye(3)[pen_state] 
        inputs = torch.cat([s, z_stack], dim=2)

    # strokes in stroke-5 format, strokes in stroke-3 format
    return to_normal_strokes(np.array(strokes))