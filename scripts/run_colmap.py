#!/usr/bin/env python3
"""
COLMAP Wrapper Script for 4DGaussians-Enhanced

This script runs COLMAP on raw image sequences to generate camera poses
and sparse point clouds for 4D Gaussian Splatting.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil


def run_command(cmd, description, check=True):
    """Run a shell command and print status."""
    print(f"🔧 {description}...")
    print(f"   Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error output:\n{result.stderr}")
        if check:
            sys.exit(1)
        return False
    
    if result.stdout:
        print(f"Output:\n{result.stdout}")
    
    print(f"✅ {description} complete")
    return True


def check_colmap_installed():
    """Check if COLMAP is installed and accessible."""
    result = subprocess.run("colmap -h", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ COLMAP not found!")
        print("\nTo install COLMAP on Colab:")
        print("  !apt-get update && apt-get install -y colmap")
        print("\nTo install COLMAP on Ubuntu/Debian:")
        print("  sudo apt-get install colmap")
        print("\nFor other systems, see: https://colmap.github.io/install.html")
        return False
    print("✅ COLMAP is installed")
    return True


def prepare_image_directory(source_path, images_dir="images"):
    """Find and prepare image directory for COLMAP."""
    source = Path(source_path)
    
    # Check if images directory already exists
    img_dir = source / images_dir
    if img_dir.exists():
        print(f"✅ Found images directory: {img_dir}")
        return str(img_dir)
    
    # Try to find images in camera folders
    cam_folders = sorted(source.glob("cam*"))
    if cam_folders:
        print(f"⚠️  Found {len(cam_folders)} camera folders")
        print("   COLMAP works best with single-view sequences")
        print("   Using first camera folder for processing")
        
        first_cam = cam_folders[0]
        frames = list(first_cam.glob("frame_*.jpg")) + list(first_cam.glob("frame_*.png"))
        
        if frames:
            print(f"   Found {len(frames)} frames in {first_cam}")
            return str(first_cam)
    
    print(f"❌ No images found in {source_path}")
    print("   Expected structure:")
    print("   • {source_path}/images/*.jpg")
    print("   • {source_path}/cam01/frame_*.jpg")
    return None


def run_colmap_pipeline(source_path, output_path, image_dir, use_gpu=True):
    """Run complete COLMAP pipeline."""
    
    source = Path(source_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    
    # Create COLMAP workspace
    colmap_dir = output / "colmap"
    colmap_dir.mkdir(exist_ok=True)
    
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    # Remove existing database
    if database_path.exists():
        print(f"🗑️  Removing existing database: {database_path}")
        database_path.unlink()
    
    # Step 1: Feature extraction
    feature_cmd = f"""colmap feature_extractor \
        --database_path {database_path} \
        --image_path {image_dir} \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model OPENCV \
        --SiftExtraction.use_gpu {'1' if use_gpu else '0'}"""
    
    run_command(feature_cmd, "Feature extraction")
    
    # Step 2: Feature matching
    matching_cmd = f"""colmap exhaustive_matcher \
        --database_path {database_path} \
        --SiftMatching.use_gpu {'1' if use_gpu else '0'}"""
    
    run_command(matching_cmd, "Feature matching")
    
    # Step 3: Sparse reconstruction (mapper)
    mapper_cmd = f"""colmap mapper \
        --database_path {database_path} \
        --image_path {image_dir} \
        --output_path {sparse_dir}"""
    
    run_command(mapper_cmd, "Sparse reconstruction")
    
    # Step 4: Find the best reconstruction
    reconstructions = list(sparse_dir.glob("*"))
    if not reconstructions:
        print("❌ No reconstructions produced by COLMAP")
        return False
    
    # Use reconstruction 0 (COLMAP's best)
    best_recon = sparse_dir / "0"
    if not best_recon.exists():
        best_recon = reconstructions[0]
    
    print(f"✅ Using reconstruction: {best_recon}")
    
    # Step 5: Convert to text format (easier to work with)
    model_converter_cmd = f"""colmap model_converter \
        --input_path {best_recon} \
        --output_path {best_recon} \
        --output_type TXT"""
    
    run_command(model_converter_cmd, "Model conversion to TXT format")
    
    print(f"\n✅ COLMAP pipeline complete!")
    print(f"📁 Output directory: {colmap_dir}")
    print(f"📁 Sparse reconstruction: {best_recon}")
    
    return True


def convert_to_4dgs_format(source_path, colmap_output):
    """Convert COLMAP output to 4DGaussians format."""
    print("\n🔄 Converting to 4DGaussians format...")
    
    # Check if conversion script exists
    converter_script = Path("scripts/colmap_converter.py")
    if not converter_script.exists():
        print("⚠️  Conversion script not found")
        print("   You may need to manually convert COLMAP output")
        print(f"   COLMAP output is at: {colmap_output}")
        return False
    
    # Run converter
    cmd = f"python {converter_script} --source_path {source_path} --colmap_path {colmap_output}"
    return run_command(cmd, "Format conversion", check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP on image sequences for 4DGaussians"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to source directory containing images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory (default: source_path/colmap_output)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="Subdirectory containing images (default: images)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip conversion to 4DGaussians format",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 COLMAP Pipeline for 4DGaussians-Enhanced")
    print("=" * 60)
    
    # Check COLMAP installation
    if not check_colmap_installed():
        sys.exit(1)
    
    # Prepare paths
    source_path = Path(args.source_path).resolve()
    if not source_path.exists():
        print(f"❌ Source path does not exist: {source_path}")
        sys.exit(1)
    
    output_path = args.output_path
    if output_path is None:
        output_path = source_path / "colmap_output"
    output_path = Path(output_path).resolve()
    
    print(f"📁 Source: {source_path}")
    print(f"📁 Output: {output_path}")
    
    # Find images
    image_dir = prepare_image_directory(source_path, args.images_dir)
    if image_dir is None:
        sys.exit(1)
    
    # Run COLMAP pipeline
    success = run_colmap_pipeline(
        str(source_path),
        str(output_path),
        image_dir,
        use_gpu=not args.no_gpu
    )
    
    if not success:
        print("\n❌ COLMAP pipeline failed")
        sys.exit(1)
    
    # Convert to 4DGaussians format
    if not args.skip_conversion:
        colmap_sparse = output_path / "colmap" / "sparse" / "0"
        convert_to_4dgs_format(str(source_path), str(colmap_sparse))
    
    print("\n" + "=" * 60)
    print("✅ COLMAP processing complete!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("  1. Verify camera poses and point cloud")
    print("  2. Generate masks with SAM2")
    print("  3. Start training with: python train.py --source_path {source_path}")


if __name__ == "__main__":
    main()
