# Semantic Segmentation with Semi-Supervised Learning

This repository implements and evaluates semi-supervised learning approaches for semantic segmentation tasks, with applications in both medical imaging and mineral identification.

## Medical Application: ACDC Dataset
![ACDC Image](doc/acdc_img.png)

**Task:** Segmentation of cardiac structures (right ventricle, myocardium, and left ventricle) from MRI scans

I reproduced and validated the results from the [UniMatch paper](https://arxiv.org/abs/2208.09910), comparing against supervised learning baselines:

| Method                | 1 labeled case | 3 labeled cases | 7 labeled cases |
|:---------------------:|:--------------:|:---------------:|:---------------:|
| UNet (Supervised)     | 28.5           | 41.5            | 62.5            |
| UniMatch (Original)   | 85.4           | 88.9            | 89.9            |
| UniMatch (Implementation) | 84.2       | 87.6            | 89.2            |

The implementation achieves near-identical performance to the original paper while requiring fewer computational resources.

## Technical Approach

This project implements several semi-supervised learning techniques:

- **FixMatch**: Incorporates pseudo-labeling with consistency regularization for self-training
- **UniMatch**: Extends FixMatch with dual consistency regularization and advanced augmentation strategies
- **Augmentation Pipeline**: Implemented CutMix, ClassMix, and feature-level perturbations

![Methods Architecture](doc/method.png)

## Mineral Segmentation Application

Applied these techniques to a real-world challenge in geological imaging analysis as part of the [Stranger Sections 2](https://thinkonward.com/app/c/challenges/stranger-sections-2) competition.

**Technical Challenge:** Segment kerogen minerals from microscopic rock slide images  
**Industry Application:** Identifying kerogens aids in locating potential natural resources such as oil and petroleum

### Dataset Visualization
![Sample image from competition dataset](doc/kerogen_img.png)
*The bright yellow patterns represent the kerogen regions to be segmented*

### Performance Results

| Method | mIoU | Improvement (from Baseline) |
|:------:|:----:|:---------------------------:|
| Baseline | 0.28 | - |
| Supervised | 0.40 | 43% |
| FixMatch | 0.46 | 64% |
| UniMatch | 0.48 | 71% |
| UniMatch (Enhanced Augmentations) | 0.49 | 75% |

## Implementation Details

- **Framework**: PyTorch
- **Hyperparameter Optimization**: Implemented Ray Tune for systematic search
- **Reference Implementation**: Based on the [original UniMatch repository](https://github.com/LiheYoung/UniMatch)

## Setup and Usage

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.10+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Segmentation-UniMatch.git
   cd Segmentation-UniMatch
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

#### ACDC Dataset
1. Download the ACDC dataset from the [official website](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. Preprocess the dataset:
   ```bash
   python scripts/preprocess_acdc.py --data_path /path/to/acdc_dataset --output_path ./data/acdc
   ```

#### Mineral Dataset
1. Download the Stranger Sections 2 dataset (if publicly available).
2. Organize the dataset following the structure:
   ```
   data/minerals/
   ├── images/
   ├── masks/
   └── splits/
       ├── labeled.txt
       └── unlabeled.txt
   ```

### Training

#### Supervised Baseline (UNet)
```bash
python train.py --config configs/supervised_acdc.yaml
```

#### Semi-Supervised (FixMatch)
```bash
python train.py --config configs/fixmatch_acdc.yaml
```

#### Semi-Supervised (UniMatch)
```bash
python train.py --config configs/unimatch_acdc.yaml --labeled_id_path data/acdc/splits/labeled_1case.txt
```

### Evaluation
```bash
python evaluate.py --config configs/eval_acdc.yaml --checkpoint path/to/checkpoint.pth
```

### Hyperparameter Tuning
```bash
python raytune_search.py --config configs/raytune_config.yaml --method unimatch --dataset acdc
```

### Inference on New Images
```bash
python predict.py --config configs/predict.yaml --input_image path/to/image.png --output_dir ./results
```

## References

- [UniMatch: A Unified Approach to Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)
- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
