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

import os
import torch
import numpy as np
from PIL import Image


def load_mask(mask_path, target_size=None):
    """
    Load binary mask from file.
    
    Args:
        mask_path (str): Path to mask file
        target_size (tuple): Optional (height, width) to resize mask to
        
    Returns:
        np.array: Binary mask with values in [0, 1]
    """
    if not os.path.exists(mask_path):
        return None
    
    mask = Image.open(mask_path).convert('L')
    
    if target_size is not None:
        mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    
    mask = np.array(mask, dtype=np.float32) / 255.0
    
    return mask


def get_mask_path_from_image_path(image_path, mask_folder="masks"):
    """
    Convert frame path to mask path.
    Format: cam01/frame_00001.jpg → cam01/masks/mask_00001.png
    
    Args:
        image_path (str): Path to image file
        mask_folder (str): Name of folder containing masks
        
    Returns:
        str: Path to corresponding mask file
    """
    # Get directory and filename
    dir_path = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    
    # Extract frame number from filename
    # Supports both frame_00001.jpg and similar patterns
    name_parts = os.path.splitext(filename)[0]
    
    # Replace "frame" with "mask" if present
    if "frame_" in name_parts:
        mask_name = name_parts.replace("frame_", "mask_")
    else:
        # If no "frame_" prefix, just use the base name
        mask_name = name_parts
    
    # Construct mask path: cam01/masks/mask_00001.png
    mask_path = os.path.join(dir_path, mask_folder, f"{mask_name}.png")
    
    return mask_path


def create_ones_mask(height, width, device="cuda"):
    """
    Create a fallback mask of all ones (no masking effect).
    
    Args:
        height (int): Mask height
        width (int): Mask width
        device (str): Device to create tensor on
        
    Returns:
        torch.Tensor: Mask of ones with shape [1, 1, H, W]
    """
    return torch.ones(1, 1, height, width, device=device)


def preprocess_mask_for_loss(mask, target_shape, device="cuda"):
    """
    Prepare mask for loss computation.
    
    Args:
        mask (np.array or torch.Tensor): Binary mask
        target_shape (tuple): Target shape (C, H, W) or (H, W)
        device (str): Device to move tensor to
        
    Returns:
        torch.Tensor: Processed mask with shape [1, 1, H, W]
    """
    if mask is None:
        # Return ones mask if no mask provided
        if len(target_shape) == 3:
            return create_ones_mask(target_shape[1], target_shape[2], device)
        else:
            return create_ones_mask(target_shape[0], target_shape[1], device)
    
    # Convert to tensor if numpy array
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # Move to device
    mask = mask.to(device)
    
    # Ensure mask has correct shape [1, 1, H, W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[0] > 1:
        # Take first channel if multi-channel
        mask = mask[0:1].unsqueeze(0)
    
    return mask
