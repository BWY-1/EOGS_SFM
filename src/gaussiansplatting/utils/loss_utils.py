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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def chromaticity_loss(network_output, gt, eps=1e-3, min_luma=0.05):
    pred = network_output.clamp_min(0.0)
    target = gt.clamp_min(0.0)

    pred_sum = pred.sum(dim=0, keepdim=True)
    target_sum = target.sum(dim=0, keepdim=True)
    pred_chroma = pred / (pred_sum + eps)
    target_chroma = target / (target_sum + eps)

    target_luma = target.mean(dim=0, keepdim=True)
    weights = ((target_luma - min_luma) / max(1.0 - min_luma, eps)).clamp(0.0, 1.0)
    weighted_diff = (pred_chroma - target_chroma).abs() * weights
    normalizer = (weights.sum() * pred.shape[0]).clamp_min(eps)
    return weighted_diff.sum() / normalizer


def global_color_loss(network_output, gt, eps=1e-3):
    pred = network_output.clamp_min(0.0)
    target = gt.clamp_min(0.0)

    pred_mean = pred.mean(dim=(1, 2))
    target_mean = target.mean(dim=(1, 2))

    pred_std = pred.std(dim=(1, 2), unbiased=False)
    target_std = target.std(dim=(1, 2), unbiased=False)

    pred_chroma = pred_mean / (pred_mean.sum(dim=0, keepdim=True) + eps)
    target_chroma = target_mean / (target_mean.sum(dim=0, keepdim=True) + eps)

    pred_std_ratio = pred_std / (pred_std.mean(dim=0, keepdim=True) + eps)
    target_std_ratio = target_std / (target_std.mean(dim=0, keepdim=True) + eps)

    mean_loss = (pred_mean - target_mean).abs().mean()
    contrast_loss = (pred_std - target_std).abs().mean()
    chroma_center_loss = (pred_chroma - target_chroma).abs().mean()
    chroma_contrast_loss = (pred_std_ratio - target_std_ratio).abs().mean()

    return (
        0.5 * mean_loss
        + 0.5 * contrast_loss
        + 1.0 * chroma_center_loss
        + 1.0 * chroma_contrast_loss
    )

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
