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
import cv2

# Optional imports (may not be available until SAM2 is installed)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Turkish to English class name aliases for YOLO
CLASS_ALIASES = {
    # Turkish -> English mappings
    "insan": "person",
    "insanlar": "person",
    "kişi": "person",
    "araba": "car",
    "otomobil": "car",
    "araç": "car",
    "köpek": "dog",
    "kedi": "cat",
    "kuş": "bird",
    "at": "horse",
    "koyun": "sheep",
    "inek": "cow",
    "fil": "elephant",
    "ayı": "bear",
    "zebra": "zebra",
    "zürafa": "giraffe",
    # Add more as needed
}


def normalize_class_name(name):
    """Normalize class name (Turkish to English, lowercase, strip)."""
    name = name.strip().lower()
    return CLASS_ALIASES.get(name, name)


class SAM2MaskGenerator:
    """
    SAM2.1-based mask generator with YOLO object detection.
    
    Pipeline:
    1. YOLO detects objects matching the prompt
    2. SAM2.1 segments each detected object
    3. Combine masks into final output
    """
    
    def __init__(self, model_size="large", device="cuda", checkpoint_dir="checkpoints"):
        """
        Initialize SAM2 mask generator.
        
        Args:
            model_size (str): Model size - "large", "base", "small", "tiny"
            device (str): Device to load model on
            checkpoint_dir (str): Directory containing SAM2.1 checkpoints and configs
        """
        self.device = device
        self.model_size = model_size
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize models
        self.sam2_predictor = None
        self.yolo_model = None
        
        self._load_sam2()
        self._load_yolo()
    
    def _load_sam2(self):
        """Load SAM2.1 model."""
        if not SAM2_AVAILABLE:
            print("❌ SAM2 not installed. Install with: pip install 'git+https://github.com/facebookresearch/sam2.git'")
            return False
        
        try:
            
            # SAM2.1 config and checkpoint mapping
            model_configs = {
                "large": ("sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
                "base": ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
                "small": ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
                "tiny": ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
            }
            
            if self.model_size not in model_configs:
                print(f"⚠️  Unknown model size {self.model_size}, using 'large'")
                self.model_size = "large"
            
            config_name, checkpoint_name = model_configs[self.model_size]
            
            # Look for config in checkpoints/configs/
            config_path = os.path.join(self.checkpoint_dir, "configs", config_name)
            if not os.path.exists(config_path):
                # Fallback: try in checkpoint_dir directly
                config_path = os.path.join(self.checkpoint_dir, config_name)
            
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            
            # Check files exist
            if not os.path.exists(config_path):
                print(f"❌ Config not found: {config_path}")
                print(f"   Run: python scripts/download_sam2.py --model-size {self.model_size}")
                return False
            
            if not os.path.exists(checkpoint_path):
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                print(f"   Run: python scripts/download_sam2.py --model-size {self.model_size}")
                return False
            
            print(f"📦 Loading SAM2.1 model: {self.model_size}")
            print(f"   Config: {config_path}")
            print(f"   Checkpoint: {checkpoint_path}")
            
            # Build SAM2 model
            sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            print(f"✅ SAM2.1 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading SAM2 model: {e}")
            return False
    
    def _load_yolo(self):
        """Load YOLO model for object detection."""
        if not YOLO_AVAILABLE:
            print("❌ ultralytics not installed. Install with: pip install ultralytics>=8.0.0")
            return False
        
        try:
            
            print("📦 Loading YOLO model...")
            # Use YOLOv8 medium for good balance of speed/accuracy
            self.yolo_model = YOLO("yolov8m.pt")
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.yolo_model.to(self.device)
            
            print("✅ YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return False
    
    def generate_mask(self, image_np, prompt="person", threshold=0.5):
        """
        Generate mask for a single image.
        
        Args:
            image_np (np.array): RGB image array (H, W, 3)
            prompt (str): Comma-separated class names (e.g., "person,human,bag")
            threshold (float): Confidence threshold for detections
            
        Returns:
            np.array: Binary mask (0 or 255), shape (H, W)
        """
        if self.sam2_predictor is None or self.yolo_model is None:
            raise RuntimeError(
                "Models not loaded properly. SAM2 or YOLO initialization failed. "
                "Check earlier error messages for details."
            )
        
        height, width = image_np.shape[:2]
        
        # Parse and normalize prompt classes
        prompt_classes = [normalize_class_name(p) for p in prompt.split(",")]
        
        # Step 1: Run YOLO detection
        results = self.yolo_model(image_np, verbose=False)
        
        if len(results) == 0:
            # No detections
            return np.zeros((height, width), dtype=np.uint8)
        
        result = results[0]
        
        # Filter detections by prompt classes and threshold
        boxes = []
        if result.boxes is not None:
            # Batch convert tensors to CPU/numpy to reduce overhead
            boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy.is_cuda else result.boxes.xyxy.numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf.is_cuda else result.boxes.conf.numpy()
            cls_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls.is_cuda else result.boxes.cls.numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, cls_ids)):
                cls_name = result.names[int(cls_id)]
                
                # Check if class matches prompt
                if cls_name in prompt_classes and float(conf) >= threshold:
                    boxes.append(box.tolist())
        
        if len(boxes) == 0:
            # No matching detections
            return np.zeros((height, width), dtype=np.uint8)
        
        # Step 2: Use SAM2 to segment each detected object
        self.sam2_predictor.set_image(image_np)
        
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for box in boxes:
            # Convert box to input format for SAM2
            input_box = np.array(box)
            
            # Predict mask
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            # Use the predicted mask (first one with multimask_output=False)
            mask = masks[0]
            
            # Add to combined mask
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
        
        # Convert to 0/255 format
        combined_mask = combined_mask * 255
        
        return combined_mask


def load_sam2_model(model_size="large", device="cuda", checkpoint_dir="checkpoints"):
    """
    Load SAM2 model for automatic mask generation.
    
    Args:
        model_size (str): Model size - "large", "base", "small", "tiny"
        device (str): Device to load model on
        checkpoint_dir (str): Directory containing SAM2 checkpoints
        
    Returns:
        SAM2MaskGenerator: Generator instance or None if loading fails
    """
    try:
        generator = SAM2MaskGenerator(
            model_size=model_size,
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        if generator.sam2_predictor is None or generator.yolo_model is None:
            return None
        
        return generator
        
    except Exception as e:
        print(f"❌ Error creating SAM2 mask generator: {e}")
        return None


def generate_masks_for_scene(source_path, mask_folder="masks", prompt="person,human", 
                            threshold=0.5, every_n=1, model_size="large", device="cuda",
                            checkpoint_dir="checkpoints"):
    """
    Batch process all cameras and frames to generate masks using SAM2.1 + YOLO.
    
    Args:
        source_path (str): Path to scene data (contains cam01, cam02, etc. or images/)
        mask_folder (str): Name of subfolder to save masks in
        prompt (str): Comma-separated detection prompts (e.g., "person,human,bag")
        threshold (float): Confidence threshold for detections
        every_n (int): Process every N frames (1 = all frames)
        model_size (str): SAM2 model size - "large", "base", "small", "tiny"
        device (str): Device to run on
        checkpoint_dir (str): Directory containing SAM2 checkpoints
        
    Returns:
        dict: Statistics about mask generation
    """
    print("🎭 Initializing SAM2.1 Mask Generator...")
    print(f"   Prompt: {prompt}")
    print(f"   Model: {model_size}")
    print(f"   Confidence threshold: {threshold}")
    print(f"   Processing every {every_n} frame(s)")
    print()
    
    # Load generator
    generator = load_sam2_model(model_size, device, checkpoint_dir)
    
    if generator is None:
        return {"error": "Failed to load SAM2 model"}
    
    # Find all camera folders or images folder
    cam_folders = sorted(glob.glob(os.path.join(source_path, "cam*")))
    
    # If no camera folders, check for images/ folder
    if len(cam_folders) == 0:
        images_folder = os.path.join(source_path, "images")
        if os.path.exists(images_folder):
            cam_folders = [images_folder]
        else:
            print(f"❌ No camera folders or images/ folder found in {source_path}")
            return {"error": "No image folders found"}
    
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
        frame_patterns = ["frame_*.jpg", "frame_*.png", "*.jpg", "*.png"]
        frame_files = []
        for pattern in frame_patterns:
            frame_files.extend(sorted(glob.glob(os.path.join(cam_folder, pattern))))
        
        # Remove duplicates (in case of overlap)
        frame_files = sorted(list(set(frame_files)))
        
        if len(frame_files) == 0:
            print(f"⚠️  No frames found in {cam_folder}")
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
        for i, frame_path in enumerate(tqdm(frame_files, desc=f"  {cam_name}", leave=False)):
            stats["total_frames"] += 1
            
            # Skip frames based on every_n
            if i % every_n != 0:
                stats["skipped_frames"] += 1
                continue
            
            try:
                # Load image
                image = Image.open(frame_path).convert("RGB")
                image_np = np.array(image)
                
                # Generate mask using SAM2 + YOLO
                mask = generator.generate_mask(image_np, prompt, threshold)
                
                # Save mask
                frame_name = os.path.basename(frame_path)
                # Generate mask name
                if "frame_" in frame_name:
                    mask_name = frame_name.replace("frame_", "mask_")
                else:
                    # For generic image names, add mask_ prefix
                    mask_name = f"mask_{frame_name}"
                mask_name = mask_name.replace(".jpg", ".png").replace(".jpeg", ".png")
                
                mask_path = os.path.join(mask_output_folder, mask_name)
                
                # Save as PNG
                Image.fromarray(mask).save(mask_path)
                
                stats["processed_frames"] += 1
                cam_stats["processed"] += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {frame_path}: {e}")
                continue
        
        stats["cameras"].append(cam_stats)
    
    print(f"\n✅ Mask generation complete!")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Processed: {stats['processed_frames']}")
    print(f"   Skipped: {stats['skipped_frames']}")
    
    return stats


def generate_mask_from_prompt(image_np, predictor, prompt, threshold=0.5):
    """
    Generate mask for a single image using text prompt.
    
    Args:
        image_np (np.array): Image array
        predictor: SAM2MaskGenerator instance
        prompt (str): Detection prompt (e.g., "person,human")
        threshold (float): Confidence threshold
        
    Returns:
        np.array: Binary mask (0 or 255)
    """
    if isinstance(predictor, SAM2MaskGenerator):
        return predictor.generate_mask(image_np, prompt, threshold)
    else:
        # Fallback: return empty mask
        print("⚠️  Invalid predictor, returning empty mask")
        return np.zeros(image_np.shape[:2], dtype=np.uint8)
