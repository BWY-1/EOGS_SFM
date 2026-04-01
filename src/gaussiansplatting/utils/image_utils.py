#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    squared_error = (img1 - img2) ** 2
    return squared_error.reshape(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse_value = mse(img1, img2).clamp_min(1e-12)
    return 20 * torch.log10(1.0 / torch.sqrt(mse_value))
