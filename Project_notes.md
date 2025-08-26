# Brain Tumor Detection Project - Research Log

## 📁 Dataset Exploration Findings

### 1. Initial Dataset Structure Discovery
**Finding:** The dataset is distributed across multiple archive folders (`archive`, `archive (1)`, ..., `archive (10)`), each containing pre-processed `.npy` files rather than raw images.

### 2. Archive Folder Structure Analysis
**Finding:** Heterogeneous organizational structures discovered across different archives:

| Archive Folder | Internal Structure | Content Type |
|----------------|-------------------|-------------|
| `archive` | `Training/`, `Testing/` subfolders | `.npy` files |
| `archive (1)` | Direct class folders (`glioma_tumor/`, `meningioma_tumor/`, `normal/`, `pituitary_tumor/`) | `.npy` files |
| `archive (2)` | Binary-style folders (`no/`, `pred/`, `yes/`) | `.npy` files |
| `archive (3)` | `Training/`, `Testing/` subfolders | `.npy` files |
| `archive (4)` | Binary classification (`no/`, `yes/`) | `.npy` files |
| `archive (5)` | Medical classification (`glioma/`, `healthy/`, `meningioma/`, `pituitary/`) | `.npy` files |
| `archive (9)` | `Training/`, `Testing/` subfolders | `.npy` files |
| `archive (10)` | ML split folders (`train/`, `valid/`) | `.npy` files |

### 3. Technical Specification
**Finding:** All data files are in NumPy array format (`.npy` extension), indicating:
- Pre-processing (resizing, normalization, noise removal) already completed
- Data ready for direct model training
- Need for custom data loading strategy

## 4. Data Specification Analysis
**Finding:** Detailed analysis of .npy files reveals:

- **Image Dimensions:** 128 × 128 pixels RGB (3 channels)
- **Data Format:** float32 with normalized values [0.0, 0.9922]
- **Pre-processing:** Complete (resizing, normalization already applied)
- **Ready for:** Direct model training without additional preprocessing

## 5. DataLoader Implementation
**Finding:** Created a memory-efficient PyTorch DataLoader that:
- Loads .npy files on-demand (prevents memory overload)
- Handles multiple archive structures automatically  
- Confirmed image dimensions: 128×128×3 (RGB)
- Ready for model training

**Test Results:** 
- Loaded 3,325 glioma samples from archive (5)
- Loaded 1,321 glioma samples from archive
- Total: 4,646 samples successfully processed

## 6. ResNet++ Model Implementation
**Achievement:** Implemented ResNet++ from scratch in pure PyTorch featuring:
- Custom residual blocks (BasicBlock)
- Channel attention mechanisms (AttentionBlock)  
- Enhanced feature extraction for medical images
- Optimized for 128×128×3 input dimensions
- 4-class output layer for tumor classification

**Test Results:** Model processes batch of 4 images → produces 4 predictions with 4 class scores each

## 🚀 Next Steps
1. Develop custom PyTorch DataLoader for NumPy arrays
2. Implement ResNet model compatible with pre-processed data
3. Create data consolidation strategy for heterogeneous structures

## 🔄 Version History
- **v0.1** - Initial project setup and repository configuration
- **v0.2** - Discovered heterogeneous archive structures
- **v0.3** - Identified exclusive use of pre-processed `.npy` files