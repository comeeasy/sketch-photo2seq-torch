# sketch-photo2seq-torch

## Introduction
> Human doesn't see world as grids of pixels.<br>
> Therefore, several methods to generate **vector** images are propoesed.<br> The seminal work, [Ha et al. 2017](https://arxiv.org/abs/1704.03477) a.k.a  `sketch-rnn` was proposed in 2017.<br>
> As a follow-up study, [Song et al. In CVPR. 2019](https://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Learning_to_Sketch_CVPR_2018_paper.pdf) a.k.a `sketch-photo2seq` was proposed to solve weak supervision problem and the [code](https://github.com/MarkMoHR/sketch-photo2seq) was provided.<br> 
> Unfortunately, The code was written in tensorflow v1.<br>
> To enjoy advances of newest version of pytorch, I produce a duplication of [sketch-photo2seq](https://github.com/MarkMoHR/sketch-photo2seq).


## Thanks to
> `sketch-photo2seq` directely borrows `sketch-rnn`'s encoder and decoder.<br> Therefore I needed pytorch version of `sketch-rnn. and directly borrow below repo.
> - [grantbey/Pytorch-SketchRNN](https://github.com/grantbey/PyTorch-SketchRNN),

> To build encoder and decoder for raster images, I directely borrowed below repo.
> - [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py)
