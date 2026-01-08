#!/usr/bin/env python3
"""
Download SAM2.1 Configuration Files and Model Checkpoints

This script downloads:
1. SAM2.1 config files from GitHub
2. SAM2.1 model checkpoints from HuggingFace
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import json


# SAM2.1 config files on GitHub
SAM2_CONFIGS = {
    "sam2.1_hiera_large": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml",
    "sam2.1_hiera_small": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
}


# SAM2.1 model checkpoints on HuggingFace
SAM2_MODELS = {
    "large": {
        "hf_repo": "facebook/sam2.1-hiera-large",
        "filename": "sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_large",
    },
    "base": {
        "hf_repo": "facebook/sam2.1-hiera-base-plus",
        "filename": "sam2.1_hiera_base_plus.pt",
        "config": "sam2.1_hiera_base_plus",
    },
    "small": {
        "hf_repo": "facebook/sam2.1-hiera-small",
        "filename": "sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_small",
    },
    "tiny": {
        "hf_repo": "facebook/sam2.1-hiera-tiny",
        "filename": "sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_tiny",
    },
}


def download_file(url, output_path, description="file"):
    """Download a file from URL to output_path."""
    try:
        print(f"📥 Downloading {description}...")
        print(f"   From: {url}")
        print(f"   To: {output_path}")
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            downloaded = blocknum * blocksize
            if totalsize > 0:
                percent = min(100, downloaded * 100 / totalsize)
                sys.stdout.write(f"\r   Progress: {percent:.1f}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print(f"\n✅ Downloaded {description}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False


def download_sam2_configs(config_dir="checkpoints/configs"):
    """Download SAM2.1 config files from GitHub."""
    print("\n" + "=" * 60)
    print("📁 Downloading SAM2.1 Config Files")
    print("=" * 60)
    
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for config_name, url in SAM2_CONFIGS.items():
        # Determine filename from URL or config name
        if "sam2.1_hiera_l.yaml" in url:
            filename = "sam2.1_hiera_l.yaml"
        elif "sam2.1_hiera_b%2B.yaml" in url:
            filename = "sam2.1_hiera_b+.yaml"
        elif "sam2.1_hiera_s.yaml" in url:
            filename = "sam2.1_hiera_s.yaml"
        elif "sam2.1_hiera_t.yaml" in url:
            filename = "sam2.1_hiera_t.yaml"
        else:
            filename = f"{config_name}.yaml"
        
        output_path = config_path / filename
        
        if output_path.exists():
            print(f"✅ Config already exists: {filename}")
            success_count += 1
            continue
        
        if download_file(url, str(output_path), f"config: {filename}"):
            success_count += 1
    
    print(f"\n✅ Downloaded {success_count}/{len(SAM2_CONFIGS)} config files")
    return success_count > 0


def download_sam2_model_hf(model_size, checkpoint_dir="checkpoints"):
    """Download SAM2.1 model from HuggingFace."""
    if model_size not in SAM2_MODELS:
        print(f"❌ Unknown model size: {model_size}")
        print(f"   Available: {list(SAM2_MODELS.keys())}")
        return False
    
    model_info = SAM2_MODELS[model_size]
    hf_repo = model_info["hf_repo"]
    filename = model_info["filename"]
    
    checkpoint_path = Path(checkpoint_dir) / filename
    
    if checkpoint_path.exists():
        print(f"✅ Model already exists: {filename}")
        return True
    
    print("\n" + "=" * 60)
    print(f"📦 Downloading SAM2.1 Model: {model_size}")
    print("=" * 60)
    
    try:
        # Use huggingface_hub to download
        from huggingface_hub import hf_hub_download
        import shutil
        
        print(f"📥 Downloading from HuggingFace: {hf_repo}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to download the model checkpoint
        # SAM2.1 models may be stored as .pt or .safetensors
        for model_filename in ["model.pt", "model.safetensors", f"{model_size}.pt"]:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=model_filename,
                    local_dir=str(checkpoint_path.parent),
                    local_dir_use_symlinks=False,
                )
                
                # Rename to expected filename
                if downloaded_path and os.path.exists(downloaded_path):
                    shutil.move(downloaded_path, str(checkpoint_path))
                    print(f"✅ Downloaded model: {filename}")
                    return True
            except Exception as e:
                # Try next filename
                continue
        
        # If we got here, none of the filenames worked
        print(f"❌ Could not find model file in {hf_repo}")
        print(f"   Tried: model.pt, model.safetensors, {model_size}.pt")
        return False
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("\n💡 Alternative: Download manually from:")
        print(f"   https://huggingface.co/{hf_repo}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download SAM2.1 configs and model checkpoints"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large",
        choices=["large", "base", "small", "tiny", "all"],
        help="Model size to download (default: large)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="checkpoints/configs",
        help="Directory to save config files (default: checkpoints/configs)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Only download config files, skip model checkpoints",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 SAM2.1 Download Tool")
    print("=" * 60)
    
    # Step 1: Download config files
    if not download_sam2_configs(args.config_dir):
        print("⚠️  Warning: Some config files failed to download")
    
    # Step 2: Download model checkpoints
    if not args.skip_models:
        if args.model_size == "all":
            for size in ["large", "base", "small", "tiny"]:
                download_sam2_model_hf(size, args.checkpoint_dir)
        else:
            download_sam2_model_hf(args.model_size, args.checkpoint_dir)
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)
    
    # Print summary
    checkpoint_path = Path(args.checkpoint_dir)
    config_path = Path(args.config_dir)
    
    if config_path.exists():
        configs = list(config_path.glob("*.yaml"))
        print(f"\n📁 Config files ({len(configs)}):")
        for cfg in sorted(configs):
            print(f"   • {cfg}")
    
    if checkpoint_path.exists():
        models = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.safetensors"))
        print(f"\n📦 Model checkpoints ({len(models)}):")
        for model in sorted(models):
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   • {model} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
