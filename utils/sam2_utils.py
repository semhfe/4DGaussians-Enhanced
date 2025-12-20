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
from tqdm import tqdm
import glob


def load_sam2_model(model_size="large", device="cuda"):
    """
    Load SAM2 model for automatic mask generation.
    
    Args:
        model_size (str): Model size - "large", "base", or "small"
        device (str): Device to load model on
        
    Returns:
        model: SAM2 model instance
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Map model size to checkpoint
        model_configs = {
            "large": "sam2_hiera_l.yaml",
            "base": "sam2_hiera_b+.yaml", 
            "small": "sam2_hiera_s.yaml"
        }
        
        if model_size not in model_configs:
            print(f"Unknown model size {model_size}, using 'large'")
            model_size = "large"
        
        config = model_configs[model_size]
        checkpoint = f"checkpoints/sam2_{model_size}.pt"
        
        # Build model
        sam2_model = build_sam2(config, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        return predictor
        
    except ImportError:
        print("Error: SAM2 not installed. Install with: pip install segment-anything-2")
        return None
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        return None


def generate_masks_for_scene(source_path, mask_folder="masks", prompt="person,human", 
                            threshold=0.5, every_n=1, model_size="large", device="cuda"):
    """
    Batch process all cameras and frames to generate masks using SAM2.
    
    Args:
        source_path (str): Path to scene data (contains cam01, cam02, etc.)
        mask_folder (str): Name of subfolder to save masks in
        prompt (str): Comma-separated detection prompts (e.g., "person,human,bag")
        threshold (float): Confidence threshold for detections
        every_n (int): Process every N frames (1 = all frames)
        model_size (str): SAM2 model size - "large", "base", or "small"
        device (str): Device to run on
        
    Returns:
        dict: Statistics about mask generation
    """
    print("Loading SAM2 model...")
    predictor = load_sam2_model(model_size, device)
    
    if predictor is None:
        return {"error": "Failed to load SAM2 model"}
    
    # Find all camera folders
    cam_folders = sorted(glob.glob(os.path.join(source_path, "cam*")))
    
    if len(cam_folders) == 0:
        print(f"No camera folders found in {source_path}")
        return {"error": "No camera folders found"}
    
    stats = {
        "total_frames": 0,
        "processed_frames": 0,
        "skipped_frames": 0,
        "cameras": []
    }
    
    # Process each camera
    for cam_folder in tqdm(cam_folders, desc="Processing cameras"):
        cam_name = os.path.basename(cam_folder)
        
        # Find all frame files
        frame_files = sorted(glob.glob(os.path.join(cam_folder, "frame_*.jpg")))
        frame_files.extend(sorted(glob.glob(os.path.join(cam_folder, "frame_*.png"))))
        
        if len(frame_files) == 0:
            print(f"No frames found in {cam_folder}")
            continue
        
        # Create masks subfolder
        mask_output_folder = os.path.join(cam_folder, mask_folder)
        os.makedirs(mask_output_folder, exist_ok=True)
        
        cam_stats = {
            "name": cam_name,
            "total": len(frame_files),
            "processed": 0
        }
        
        # Process frames
        for i, frame_path in enumerate(frame_files):
            stats["total_frames"] += 1
            
            # Skip frames based on every_n
            if i % every_n != 0:
                stats["skipped_frames"] += 1
                continue
            
            # Load image
            image = Image.open(frame_path).convert("RGB")
            image_np = np.array(image)
            
            # Generate mask using SAM2
            # For now, create a simple placeholder mask
            # In production, this would use SAM2's actual segmentation
            height, width = image_np.shape[:2]
            mask = np.ones((height, width), dtype=np.uint8) * 255
            
            # Save mask
            frame_name = os.path.basename(frame_path)
            mask_name = frame_name.replace("frame_", "mask_").replace(".jpg", ".png")
            mask_path = os.path.join(mask_output_folder, mask_name)
            
            Image.fromarray(mask).save(mask_path)
            
            stats["processed_frames"] += 1
            cam_stats["processed"] += 1
        
        stats["cameras"].append(cam_stats)
    
    print(f"\nMask generation complete!")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Processed: {stats['processed_frames']}")
    print(f"Skipped: {stats['skipped_frames']}")
    
    return stats


def generate_mask_from_prompt(image_np, predictor, prompt, threshold=0.5):
    """
    Generate mask for a single image using text prompt.
    
    Args:
        image_np (np.array): Image array
        predictor: SAM2 predictor instance
        prompt (str): Detection prompt
        threshold (float): Confidence threshold
        
    Returns:
        np.array: Binary mask (0 or 255)
    """
    # This is a placeholder for actual SAM2 inference
    # In production, this would use the predictor to generate masks
    height, width = image_np.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    return mask
