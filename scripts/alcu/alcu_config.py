"""NCSN++ / VE-SDE configuration for 384x384 single-channel Al-Cu phi patches."""
from __future__ import annotations
import torch
import ml_collections

def get_config(batch_size: int = 2, n_iters: int = 300000, image_size: int = 384):
    config = ml_collections.ConfigDict()
    training = config.training = ml_collections.ConfigDict()
    training.sde = "vesde"
    training.continuous = True
    training.batch_size = int(batch_size)
    training.n_iters = int(n_iters)
    training.likelihood_weighting = False
    training.reduce_mean = False

    data = config.data = ml_collections.ConfigDict()
    data.dataset = "ALCU_PHASE_FIELD"
    data.image_size = int(image_size)
    data.centered = False
    data.num_channels = 1
    data.uniform_dequantization = False
    data.random_flip = False

    model = config.model = ml_collections.ConfigDict()
    model.name = "ncsnpp"
    model.sigma_max = 378.0
    model.sigma_min = 0.01
    model.num_scales = 2000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "residual"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3

    optim = config.optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 0.0
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 42
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return config
