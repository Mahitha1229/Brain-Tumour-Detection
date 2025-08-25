"""
Custom DataLoader for NPY files
Purpose: Create PyTorch DataLoader from pre-processed .npy files
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class NumpyDataset(Dataset):
    def __init__(self, archive_path, class_name='glioma'):
        self.data = []
        
        # Convert to Path object and check if path exists
        base_path = Path(archive_path)
        if not base_path.exists():
            raise ValueError(f"Archive path not found: {archive_path}")
        
        # SMART PATH DETECTION: Try different possible structures
        possible_paths = [
            base_path / class_name,                    # archive (5)/glioma/
            base_path / 'Training' / class_name,       # archive/Training/glioma/
            base_path / 'train' / class_name,          # archive (10)/train/glioma/
            base_path / (class_name + '_tumor')         # archive (1)/glioma_tumor/
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            print(f"Error: Could not find {class_name} folder in {archive_path}")
            print("Available folders:", [item.name for item in base_path.iterdir() if item.is_dir()])
            return
        
        # Load all .npy files from the directory
        npy_files = list(data_path.glob("*.npy"))
        if not npy_files:
            print(f"Warning: No .npy files found in {data_path}")
            return
            
        for file_path in npy_files:
            self.data.append(np.load(file_path))
        
        # Convert list of arrays to single tensor
        self.data = torch.from_numpy(np.array(self.data)).float()
        print(f"✓ Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and dummy label (we'll add real labels later)
        image = self.data[idx]
        label = 0  # Temporary dummy label
        return image, label

# Test the DataLoader
if __name__ == "__main__":
    print("Testing NumpyDataset DataLoader...")
    
    # Test with archive (5) which has direct class folders
    print("\n1. Testing archive (5) with direct class folders:")
    dataset = NumpyDataset("archive (5)", class_name='glioma')
    
    if len(dataset) > 0:
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test one batch
        for images, labels in dataloader:
            print(f"Batch images shape: {images.shape}")
            print(f"Batch labels: {labels}")
            break
    
    # Also test with regular archive structure
    print("\n2. Testing archive with Training/Testing structure:")
    dataset2 = NumpyDataset("archive", class_name='glioma')