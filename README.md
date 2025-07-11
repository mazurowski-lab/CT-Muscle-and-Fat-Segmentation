# CT Muscle and Fat Segmentation for Skeletal Muscle, SAT, VAT, and Muscular Fat

**By [Yaqian Chen](https://scholar.google.com/citations?user=iegKFuQAAAAJ&hl=en), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en&oi=ao), [Yuwen Chen](https://scholar.google.com/citations?user=61s49p0AAAAJ&hl=en&oi=ao), [Jicheng Yang](https://scholar.google.com/citations?user=jGv3bRUAAAAJ&hl=en&oi=ao), [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en&oi=ao), [Joseph Y. Cao](#), [Adrian Camarena](#), [Christopher Mantyh](#), [Roy Colglazier](#), and [Maciej Mazurowski](https://scholar.google.com/citations?user=HlxjJPQAAAAJ&hl=en&oi=ao)**

Please email [yaqian.chen@duke.edu](mailto:yaqian.chen@duke.edu) for any problem with this code.
---
[![arXiv](https://img.shields.io/badge/arXiv-2502.09779-b31b1b.svg)](https://arxiv.org/abs/2502.09779)

Download the required pre-trained model weights from [this Google Drive folder](https://drive.google.com/drive/folders/1q3_v4a-P9hIAphnOhhuSoL5FcJ7IIsDX?usp=sharing). Make sure to select the appropriate files for your configuration.

This is the official code for our paper:  
**Automated Muscle and Fat Segmentation in Computed Tomography for Comprehensive Body Composition Analysis**

<img width="1000" alt="Screenshot 2025-02-13 at 3 17 37 AM" src="https://github.com/user-attachments/assets/6a5db7af-a005-4779-82e3-4d5242ba12d1" />

where we developed an open-source method for segmenting skeletal muscle, subcutaneous adipose tissue (SAT), and visceral adipose tissue (VAT) across the chest, abdomen, and pelvis in axial CT images.This method provides various body composition metrics, including muscle density, visceral-to-subcutaneous fat (VAT/SAT) ratio, muscle area/volume, and skeletal muscle index (SMI). It supports both 2D assessments at the L3 level and 3D assessments spanning from T12 to L4.

---
Below is the tutorial that explains how to use our automated muscle and fat segmentation tool.  
## 1) Installation
1. First, create a new conda environment:
```python
conda create -n muscle_seg python=3.12
conda activate muscle_se
```
2. Install required packages:
```python
pip install -r requirements.txt
```
## Script Overview

The script (`predict_muscle_fat.py`) performs three main tasks:
1. Muscle and fat segmentation using nnUNet
2. Vertebrae detection using TotalSegmentator
3. Body composition analysis (2D at L3 level and/or 3D between T12-L4)

## Usage

### Basic Command
```bash
python predict_muscle_fat.py --input <input_path> --output <output_path>
```

### Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input NIfTI file or folder | "demo" |
| `--output` | Path for saving segmentation results | "results" |
| `--save_probabilities` | Save probability maps | False |
| `--overwrite` | Overwrite existing predictions | False |
| `--checkpoint_type` | Use 'best' or 'final' checkpoint | "final" |
| `--body_composition_type` | Choose '2D', '3D', 'both', or 'None' | "both" |
| `--output_body_composition` | Path for body composition metrics | None |

### Examples

1. Process a single file:
```bash
python predict_muscle_fat.py --input patient001.nii.gz --output results/
```

2. Process a directory of images:
```bash
python predict_muscle_fat.py --input dataset/images/ --output results/ --body_composition_type both
```

3. Use best checkpoint and save probabilities:
```bash
python predict_muscle_fat.py --input dataset/images/ --output results/ --checkpoint_type best --save_probabilities True
```
## Output Structure

### Directory Structure
```
output_path/
├── input_name_segmentation.nii.gz    # Muscle/fat segmentation
└── input_name_vertebrae_segmentation/  # Vertebrae segmentation
    ├── vertebrae_T12.nii.gz
    ├── vertebrae_L3.nii.gz
    └── vertebrae_L4.nii.gz
```

### Body Composition Files
```
output_path/
├── body_composition_2d.csv   # L3-level metrics
└── body_composition_3d.csv   # T12-L4 metrics
```

## Metrics

### 2D Metrics (at L3)
- `muscle_area_mm2`: Muscle area at L3 level
- `muscle_density_hu`: Average muscle density in Hounsfield units
- `sfat_area_mm2`: Subcutaneous fat area
- `vfat_area_mm2`: Visceral fat area
- `mfat_area_mm2`: Muscle fat area
- `total_fat_area_mm2`: Total fat area
- `body_area_mm2`: Total body area

### 3D Metrics (T12-L4)
- `muscle_volume_mm3`: Muscle volume
- `muscle_density_hu`: Average muscle density
- `sfat_volume_mm3`: Subcutaneous fat volume
- `vfat_volume_mm3`: Visceral fat volume
- `mfat_volume_mm3`: Muscle fat volume
- `total_fat_volume_mm3`: Total fat volume
- `body_volume_mm3`: Total body volume
- `body_height_mm`: Height between T12 and L4

## Tips and Troubleshooting
### Input Data Requirements
- Use NIfTI format (.nii.gz)
- Ensure proper orientation (typically axial)
- Check image quality and contrast
- Check image direction (align with TotalSegmentator requirement)
### Common Issues
| Issue | Solution |
|-------|----------|
| Missing vertebrae | Check image coverage |
| Vertebrae segmentation errors | Verify image orientation |

This tool combines state-of-the-art deep learning models for accurate muscle and fat analysis in medical images. For more information, please refer to the documentation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
