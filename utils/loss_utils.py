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
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

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

def l1_loss_masked(network_output, gt, mask, w_fg=1.0, w_bg=0.1):
    """
    Masked L1 loss with separate foreground/background weights.
    
    Args:
        network_output (torch.Tensor): Predicted image [B, C, H, W]
        gt (torch.Tensor): Ground truth image [B, C, H, W]
        mask (torch.Tensor): Binary mask [B, 1, H, W] where 1=foreground, 0=background
        w_fg (float): Weight for foreground loss
        w_bg (float): Weight for background loss
        
    Returns:
        torch.Tensor: Weighted masked L1 loss
    """
    # Compute per-pixel L1 loss
    diff = torch.abs(network_output - gt)
    
    # Apply mask weights: foreground gets w_fg, background gets w_bg
    weights = mask * w_fg + (1 - mask) * w_bg
    
    # Compute weighted loss
    weighted_loss = diff * weights
    
    return weighted_loss.mean()

def ssim_masked(img1, img2, mask, window_size=11):
    """
    Masked SSIM - compute SSIM with mask weighting.
    
    Args:
        img1 (torch.Tensor): First image [B, C, H, W]
        img2 (torch.Tensor): Second image [B, C, H, W]
        mask (torch.Tensor): Binary mask [B, 1, H, W] where 1=foreground
        window_size (int): Size of SSIM window
        
    Returns:
        torch.Tensor: Masked SSIM value
    """
    # Small epsilon to prevent division by zero
    EPSILON = 1e-8
    
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Compute SSIM map
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
    
    # Apply mask weighting (mask is per-channel, expand to match SSIM map)
    if mask is not None:
        # Expand mask to match number of channels if needed
        if mask.size(1) == 1 and ssim_map.size(1) > 1:
            mask = mask.expand(-1, ssim_map.size(1), -1, -1)
        
        # Weight SSIM map by mask
        masked_ssim = (ssim_map * mask).sum() / (mask.sum() + EPSILON)
        return masked_ssim
    else:
        return ssim_map.mean()

