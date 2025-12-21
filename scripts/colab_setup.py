#!/usr/bin/env python3
"""
Colab Setup Script for 4DGaussians-Enhanced

This script handles:
1. C++ compilation fixes (AT_CHECK → TORCH_CHECK, add cfloat includes)
2. Ninja installation for faster compilation
3. SAM2 installation from GitHub
"""

import os
import sys
import subprocess
import glob


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"🔧 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    print(f"✅ {description} complete")
    return True


def install_ninja():
    """Install ninja build system for faster compilation."""
    print("\n📦 Installing Ninja build system...")
    if run_command("pip install ninja", "Ninja installation"):
        print("✅ Ninja installed successfully")
        return True
    return False


def patch_cpp_files():
    """Patch C++ files in submodules to fix PyTorch 2.x compatibility."""
    print("\n🔧 Patching C++ files for PyTorch 2.x compatibility...")
    
    # Find all .cpp, .cu, and .h files in submodules
    file_patterns = [
        "submodules/**/*.cpp",
        "submodules/**/*.cu",
        "submodules/**/*.h",
        "submodules/**/*.cuh"
    ]
    
    files_to_patch = []
    for pattern in file_patterns:
        files_to_patch.extend(glob.glob(pattern, recursive=True))
    
    if not files_to_patch:
        print("⚠️  No C++ files found in submodules (they may not be initialized yet)")
        print("    Run 'git submodule update --init --recursive' first")
        return True
    
    patch_count = 0
    
    for filepath in files_to_patch:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # Patch 1: Replace AT_CHECK with TORCH_CHECK
            if 'AT_CHECK' in content:
                content = content.replace('AT_CHECK', 'TORCH_CHECK')
                modified = True
                print(f"  • {filepath}: AT_CHECK → TORCH_CHECK")
            
            # Patch 2: Add #include <cfloat> if FLT_MAX is used but cfloat not included
            if 'FLT_MAX' in content or 'FLT_MIN' in content:
                if '#include <cfloat>' not in content and '#include<cfloat>' not in content:
                    # Find a good place to add the include (after other includes)
                    lines = content.split('\n')
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('#include'):
                            insert_idx = i + 1
                    
                    if insert_idx > 0:
                        lines.insert(insert_idx, '#include <cfloat>')
                        content = '\n'.join(lines)
                        modified = True
                        print(f"  • {filepath}: Added #include <cfloat>")
            
            # Write back if modified
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                patch_count += 1
        
        except Exception as e:
            print(f"⚠️  Warning: Could not patch {filepath}: {e}")
    
    if patch_count > 0:
        print(f"✅ Patched {patch_count} files")
    else:
        print("✅ No patches needed")
    
    return True


def install_sam2_from_github():
    """Install SAM2 from GitHub repository."""
    print("\n📦 Installing SAM2 from GitHub...")
    
    # Install SAM2 from GitHub
    cmd = 'pip install "git+https://github.com/facebookresearch/sam2.git"'
    if run_command(cmd, "SAM2 installation from GitHub"):
        print("✅ SAM2 installed successfully from GitHub")
        return True
    
    print("❌ Failed to install SAM2 from GitHub")
    return False


def verify_installation():
    """Verify that key packages are installed."""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import sam2
        print("✅ SAM2 installed")
    except ImportError:
        print("⚠️  SAM2 not found (will be installed)")
    
    try:
        import ninja
        print("✅ Ninja installed")
    except ImportError:
        print("⚠️  Ninja not found (will be installed)")
    
    return True


def main():
    """Main setup routine."""
    print("=" * 60)
    print("🚀 4DGaussians-Enhanced Colab Setup")
    print("=" * 60)
    
    # Step 1: Verify current installation
    verify_installation()
    
    # Step 2: Install Ninja
    install_ninja()
    
    # Step 3: Install SAM2 from GitHub
    install_sam2_from_github()
    
    # Step 4: Patch C++ files
    patch_cpp_files()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("  1. Run: git submodule update --init --recursive")
    print("  2. Run: pip install -e submodules/depth-diff-gaussian-rasterization")
    print("  3. Run: pip install -e submodules/simple-knn")
    print("  4. Download SAM2 configs and models with scripts/download_sam2.py")


if __name__ == "__main__":
    main()
