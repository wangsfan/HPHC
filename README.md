# Hierarchical prototypes consolidate the hierarchical consistency of knowledge in weakly-supervised incremental segmentation (HPHC)

 ## Overview

The code implements weakly-supervised incremental segmentation with Swin Transformer backbone, using the hierarchical prototpyes. 


## Project structure

```

├── Swin-Transformer-Semantic-Segmentation/   # mmsegmentation based Swin Transformer segmantation backbone
│   ├── configs/swin/                         # Swin Transformer configuration file
│   ├── mmseg/                                # mmsegmentation library
│   ├── tools/                                # Training and test script
│   └── checkpoints/                          # Directory of pre-trained weights
├── WILSON/                                   # WILSON data load and incremental learning tool
│   └── data/                                 # Dataset storage directory
│       └── voc/                              # PASCAL VOC dataset
├── train_mmseg_swin_large_voc_15_5_step0_saved.py        # Step0 fully-supervised（Basic task, 15 classes）
├── train_mmseg_swin_large_voc_15_5_step0_saved_weakly.py # Step0 weakly-supervised（with pseudo labels）
├── train_mmseg_swin_large_voc_15_5_step1.py              # Step1 Incremental session（new 5 classes）
├── train_mmseg_swin_large_voc_15_5.py                    # 15-5 scenario training 
├── train_mmseg_swin_large_voc_saved.py                   # full class training script
├── train_mmseg_swin_large_pseudo_labels.py               # using SIPE pseudo label for training
├── test_mmseg_swin_large_voc_15_5_step0.py               # Step0 model testing
├── test_mmseg_swin_large_voc_21class.py                  # full class（21 classes）model testing
├── test_mmseg_swin_tiny_voc_15_5_step1.py                # Step1 model testing（Swin-Tiny）
├── load_voc_dataset.py                                    # VOC dataset loading tool
└── README.md
```

## Environment configuration

### 1. Basic environment

- Python 3.7.0
- PyTorch 1.7.0
- CUDA 11.0
- NVIDIA GeForce RTX 3090 GPU (single)

### 2. Installation dependency

```bash
# create and activate conda 
conda create -n mst python=3.7 -y
conda activate mst

# Install PyTorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# Install mmcv
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# Install mmsegmentation
cd Swin-Transformer-Semantic-Segmentation
pip install -e .

# Installation dependency
pip install timm matplotlib terminaltables numpy tqdm pillow
pip install swanlab  # optional, for experiment tracing
```

## Data preparation

### PASCAL VOC 2012

1. Download PASCAL VOC 2012 dataset and SBD extension annotation, and save in  `WILSON/data/voc/` directory

```
WILSON/data/voc/
├── JPEGImages/              # Original images（10,582 for training, 1,449 for validation）
├── SegmentationClassAug/    # Pixel-level segmentation labels (with SBD extension)
├── splits/
│   ├── train_aug.txt        # Training set file list
│   └── val.txt              # Validation set file list
├── voc_1h_labels_train.npy  # Image-level labels (one-hot form, for weak supervision)
├── voc_1h_labels_val.npy
└── 15-5/                    # 15-5 Scenario data partition
    ├── train-0.npy          # Step0 Training set index
    ├── train-1.npy          # Step1 Training set index
    ├── val-0.npy
    └── val-1.npy
```

### Pre-training weight

Download Swin Transformer pre-training weights, and save in `Swin-Transformer-Semantic-Segmentation/checkpoints/`

```bash
cd Swin-Transformer-Semantic-Segmentation/checkpoints/

# Swin-Tiny (Recommend)
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

# Swin-Large (Optional, larger)
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
```

## WSIS scenario instruction

WSIS utilizes Task number T=2

| Scenario | Basic session (Step0) | Incremental session (Step1) | Supervision fashion |
|------|------------------|------------------|----------|
| VOC-15-5 | 15 classes + background（pixel-level annotation） | New 5 classes (Image-level labels) 
| VOC-10-10 | 10 classes + background（pixel-level annotation） | New 10 classes (Image-level labels)

- **D (Disjoint) scenario**: Without old class instances in incremental session
- **O (Overlap) scenario**: With old class instances in incremental session

## Training flowchart

The training process consists of two steps.

### Step0: Basic session (full supervision)

In basic session, we use pixel-level annotation to train Swin Transformer + UPerNet 

```bash
python train_mmseg_swin_large_voc_15_5_step0_saved.py
```

Main hyperparameters: 
- Backbone: Swin-Tiny（embed_dim=96, depths=[2,2,6,2], window_size=7）
- Segmentation head: UPerNet
- Optimizer: AdamW（lr=6e-5, weight_decay=0.01）
- Learning rate: Poly strategy, with Warmup 1500 epochs
- Input size: 512 x 512
- Batch Size: 8
- Max epoch number: 50 epochs

After training, the model is saved in `outputs/mmseg_swin_tiny_voc_15-5_step0/` 

### Step1: Incremental session training (Weak supervision)

Load Step0 model, expand classifier head to 21 classes, and use image-level labels to train new classes: 

```bash
python train_mmseg_swin_large_voc_15_5_step1.py
```

Key mechanism in incremental session: 
1. **Hierarchical prototype modeling**: by integrating hierarhical feature with a spatial-and-class dual similarity modeling strategy
2. **Fine-grained CAM construction**: by purifying high-resolution semantics under the guidance of hierarchical prototpyes

Loss functions: 
- `L_CLS`: Multi-label classification loss
- `L_CFC`: Coarse-to-Fine consistency loss (preserving consistency of coarse and fine-level CAMs)
- `L_HPC`: Hierarchical Prototype contrastive loss (separating prototypes)
- `L_SEG`: Segmentation loss (for segmentation head of pseudo-label supervision) 

Total loss: `L_total = L_CLS + L_CFC + L_SEG + α·L_HPC`, with α=0.5

### Weakly-supervised training with SIPE pseudo labels

Use SIPE pseudo labels for training

```bash
python train_mmseg_swin_large_voc_15_5_step0_saved_weakly.py
```

Or use pseudo label for training all the 21 classes

```bash
python train_mmseg_swin_large_pseudo_labels.py
```

## Test and Evaluation

### Test Step0 Model

```bash
python test_mmseg_swin_large_voc_15_5_step0.py
```

### Test Step1 Model （After incremental session）

```bash
python test_mmseg_swin_tiny_voc_15_5_step1.py
```

### Test all-class segmentation

```bash
python test_mmseg_swin_large_voc_21class.py
```

Evaluation metric（mIoU）
- Old class mIoU (stability)
- New class mIoU (plasticity)
- All class mIoU (overall performance)

## Model configuration

Swin Transformer configuration file is in `Swin-Transformer-Semantic-Segmentation/configs/swin/`, used for

| Configuration | instruction |
|---------|------|
| `upernet_swin_tiny_patch4_window7_512x512_20k_voc12aug.py` | Swin-Tiny + UPerNet（recommend） |
| `upernet_swin_large_patch4_window7_512x512_40k_voc12aug.py` | Swin-Large + UPerNet |

## Experiment tracing

The project supports [SwanLab](https://swanlab.cn/) 

## Reference

- Swin Transformer: [Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021]
- UPerNet: [Xiao et al., "Unified Perceptual Parsing for Scene Understanding", ECCV 2018]
- WILSON: [Kim et al., "Weakly Incremental Learning for Semantic Segmentation", CVPR 2023]
- SIPE: [Chen et al., "Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation", CVPR 2022]
- mmsegmentation: [OpenMMLab Semantic Segmentation Toolbox]
