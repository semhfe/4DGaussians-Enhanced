#!/usr/bin/env python3
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

"""
Temporal smoothing to reduce mask flickering between frames.
Uses temporal averaging to smooth mask transitions.

Usage:
    python scripts/smooth_masks.py --source_path /path/to/scene --mask_folder masks --window_size 3
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob


def smooth_masks_temporal(mask_paths, output_paths, window_size=3, threshold=0.5):
    """
    Apply temporal smoothing to a sequence of masks.
    
    Args:
        mask_paths (list): List of paths to input masks
        output_paths (list): List of paths to save smoothed masks
        window_size (int): Size of temporal averaging window
        threshold (float): Threshold for binarizing smoothed masks
    """
    if len(mask_paths) == 0:
        return
    
    # Load all masks
    masks = []
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        masks.append(mask_np)
    
    masks = np.array(masks)  # Shape: (T, H, W)
    
    half_window = window_size // 2
    smoothed_masks = np.zeros_like(masks)
    
    # Apply temporal averaging
    for t in range(len(masks)):
        start_t = max(0, t - half_window)
        end_t = min(len(masks), t + half_window + 1)
        
        # Average masks in window
        smoothed_masks[t] = masks[start_t:end_t].mean(axis=0)
    
    # Binarize and save
    for t, output_path in enumerate(output_paths):
        smoothed = (smoothed_masks[t] > threshold).astype(np.uint8) * 255
        Image.fromarray(smoothed).save(output_path)


def smooth_masks_in_scene(source_path, mask_folder="masks", output_suffix="_smooth",
                         window_size=3, threshold=0.5):
    """
    Apply temporal smoothing to all masks in a scene.
    
    Args:
        source_path (str): Path to scene data
        mask_folder (str): Name of mask subfolder
        output_suffix (str): Suffix to add to output folder name
        window_size (int): Size of temporal averaging window
        threshold (float): Threshold for binarizing smoothed masks
    """
    # Find all camera folders
    cam_folders = sorted(glob.glob(os.path.join(source_path, "cam*")))
    
    if len(cam_folders) == 0:
        print(f"No camera folders found in {source_path}")
        return
    
    total_masks = 0
    
    for cam_folder in tqdm(cam_folders, desc="Processing cameras"):
        mask_input_folder = os.path.join(cam_folder, mask_folder)
        
        if not os.path.exists(mask_input_folder):
            print(f"Mask folder not found: {mask_input_folder}")
            continue
        
        # Create output folder (either overwrite or create new)
        if output_suffix == "":
            mask_output_folder = mask_input_folder
        else:
            mask_output_folder = os.path.join(cam_folder, mask_folder + output_suffix)
            os.makedirs(mask_output_folder, exist_ok=True)
        
        # Find all mask files
        mask_files = sorted(glob.glob(os.path.join(mask_input_folder, "*.png")))
        
        if len(mask_files) == 0:
            continue
        
        # Prepare output paths
        output_paths = [os.path.join(mask_output_folder, os.path.basename(f)) 
                       for f in mask_files]
        
        # Apply temporal smoothing
        smooth_masks_temporal(mask_files, output_paths, window_size, threshold)
        total_masks += len(mask_files)
    
    print(f"\nSmoothed {total_masks} masks")
    
    if output_suffix != "":
        print(f"Smoothed masks saved to: {mask_folder}{output_suffix}")
    else:
        print(f"Masks smoothed in place")


def main():
    parser = argparse.ArgumentParser(description="Apply temporal smoothing to masks")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to scene data containing camera folders")
    parser.add_argument("--mask_folder", type=str, default="masks",
                       help="Name of mask subfolder (default: masks)")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Suffix for output folder. Empty string overwrites in place (default: '')")
    parser.add_argument("--window_size", type=int, default=3,
                       help="Size of temporal averaging window (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binarizing smoothed masks (default: 0.5)")
    parser.add_argument("--inplace", action="store_true",
                       help="Overwrite masks in place (sets output_suffix to '')")
    
    args = parser.parse_args()
    
    if args.inplace:
        args.output_suffix = ""
    
    print(f"Smoothing masks in: {args.source_path}")
    print(f"Mask folder: {args.mask_folder}")
    print(f"Window size: {args.window_size}")
    print(f"Threshold: {args.threshold}")
    
    if args.output_suffix == "":
        response = input("This will overwrite existing masks. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    smooth_masks_in_scene(args.source_path, args.mask_folder, args.output_suffix,
                         args.window_size, args.threshold)


if __name__ == "__main__":
    main()
