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
Fill holes inside foreground masks using morphological operations.
Useful for cleaning up masks with small gaps.

Usage:
    python scripts/fill_mask_holes.py --source_path /path/to/scene --mask_folder masks --kernel_size 5
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import cv2


def fill_mask_holes(mask_path, output_path, kernel_size=5, iterations=1):
    """
    Fill holes in a single mask using morphological closing.
    
    Args:
        mask_path (str): Path to input mask
        output_path (str): Path to save filled mask
        kernel_size (int): Size of morphological kernel
        iterations (int): Number of closing iterations
    """
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological closing to fill holes
    filled = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    Image.fromarray(filled).save(output_path)


def fill_holes_in_scene(source_path, mask_folder="masks", output_suffix="_filled", 
                        kernel_size=5, iterations=1):
    """
    Fill holes in all masks in a scene.
    
    Args:
        source_path (str): Path to scene data
        mask_folder (str): Name of mask subfolder
        output_suffix (str): Suffix to add to output folder name
        kernel_size (int): Size of morphological kernel
        iterations (int): Number of closing iterations
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
        
        for mask_path in mask_files:
            mask_name = os.path.basename(mask_path)
            output_path = os.path.join(mask_output_folder, mask_name)
            
            fill_mask_holes(mask_path, output_path, kernel_size, iterations)
            total_masks += 1
    
    print(f"\nFilled holes in {total_masks} masks")
    
    if output_suffix != "":
        print(f"Filled masks saved to: {mask_folder}{output_suffix}")
    else:
        print(f"Masks filled in place")


def main():
    parser = argparse.ArgumentParser(description="Fill holes in binary masks")
    parser.add_argument("--source_path", type=str, required=True,
                       help="Path to scene data containing camera folders")
    parser.add_argument("--mask_folder", type=str, default="masks",
                       help="Name of mask subfolder (default: masks)")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Suffix for output folder. Empty string overwrites in place (default: '')")
    parser.add_argument("--kernel_size", type=int, default=5,
                       help="Size of morphological kernel (default: 5)")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of closing iterations (default: 1)")
    parser.add_argument("--inplace", action="store_true",
                       help="Overwrite masks in place (sets output_suffix to '')")
    
    args = parser.parse_args()
    
    if args.inplace:
        args.output_suffix = ""
    
    print(f"Filling holes in masks: {args.source_path}")
    print(f"Mask folder: {args.mask_folder}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Iterations: {args.iterations}")
    
    if args.output_suffix == "":
        response = input("This will overwrite existing masks. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    fill_holes_in_scene(args.source_path, args.mask_folder, args.output_suffix,
                       args.kernel_size, args.iterations)


if __name__ == "__main__":
    main()
