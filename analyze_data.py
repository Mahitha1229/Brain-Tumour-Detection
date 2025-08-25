"""
NPY File Structure Analysis Script
Purpose: Analyze the structure and content of pre-processed .npy files
"""

import numpy as np
import os
from pathlib import Path

def analyze_npy_file(file_path):
    """Analyze a single .npy file and print its properties"""
    try:
        print(f"\n🔍 Analyzing: {file_path}")
        data = np.load(file_path)
        
        print(f"   Shape: {data.shape}")
        print(f"   Data type: {data.dtype}")
        print(f"   Value range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"   Dimensions: {data.ndim}D")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return False

def explore_directory_structure():
    """Explore the structure of archive directories"""
    print("📁 Directory Structure Exploration")
    print("=" * 50)
    
    archive_folders = [f"archive ({i})" for i in range(1, 11)] + ["archive"]
    
    for archive in archive_folders:
        archive_path = Path(archive)
        if archive_path.exists():
            print(f"\n📂 {archive}:")
            items = list(archive_path.iterdir())
            for item in items[:5]:  # Show first 5 items
                print(f"   ├── {item.name} {'(dir)' if item.is_dir() else '(file)'}")
            if len(items) > 5:
                print(f"   └── ... and {len(items) - 5} more items")

if __name__ == "__main__":
    print("Brain Tumor Detection - Data Analysis")
    print("=" * 50)
    
    # First, explore directory structures
    explore_directory_structure()
    
    # Then try to analyze a sample file from a known archive
    sample_paths = [
        "archive (5)/glioma/Fernando_Feltrin_glioma_100_jpg.npy",  # Use the actual filename
        "archive (5)/glioma/Fernando_Feltrin_glioma_101_jpg.npy",
        "archive/Training/glioma/Tr-me_0000.npy"
    ]
    
    print("\n\n🔬 File Content Analysis")
    print("=" * 50)
    
    for path in sample_paths:
        if analyze_npy_file(path):
            break  # Stop after first successful analysis
    else:
        print("❌ Could not find any sample files with common names")
        print("💡 Please manually explore one archive folder and provide a full file path")