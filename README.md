# HW4 - Visual Recognition using Deep Learning (Image Restoration with PromptIR)

Titouan GAGNEUX, ID : 114550830

## Introduction

This project is part of the *Visual Recognition using Deep Learning* course (Spring 2026).

The objective of this homework is to solve an **image restoration problem** involving two different degradation types:

* Rain degradation
* Snow degradation

Unlike a standard image classification task, image restoration requires the model to reconstruct a clean high-quality image from a degraded input image while preserving textures, structures, edges, and colors.

The dataset contains:

* 1600 rain training pairs
* 1600 snow training pairs
* 100 degraded test images

The evaluation metric used for the competition is:

* **PSNR (Peak Signal-to-Noise Ratio)**

The homework constraints are:

* Use **PromptIR** as the restoration model
* Train a **single model** capable of restoring both rain and snow images
* No external data allowed
* No pretrained weights allowed

The goal is to achieve the highest possible PSNR score on the competition leaderboard while respecting all assignment constraints.


---

## Environment Setup

### Requirements

* Python >= 3.9
* PyTorch
* torchvision
* torch
* scikit-learn
* NumPy
* OpenCV
* matplotlib
* tqdm
* Pillow

### Installation

```bash
pip install -r requirements.txt
```

---

## Method Overview

### Dataset Preprocessing

The dataset contains paired degraded and clean RGB images.

The preprocessing pipeline includes:

* Automatic matching between degraded and clean image pairs
* Train / validation split
* Tensor normalization to [0,1]
* Random patch extraction for memory-efficient training

Patch-based training was used because restoration networks are memory-intensive when processing large images.

### Data Augmentation

Paired augmentations were applied to both degraded and clean images simultaneously:

* Random horizontal flipping
* Random vertical flipping
* Random rotations

The augmentations were intentionally kept moderate in order to preserve realistic degradation patterns.

### Model Architecture

The proposed model is based on an improved PromptIR-inspired architecture.

Main characteristics:

* Lightweight encoder-decoder restoration network
* Residual convolutional blocks
* Multi-scale feature extraction
* Skip connections
* PromptIR-inspired conditioning mechanisms

The final model contains approximately:

* 1.64M trainable parameters

The architecture was improved compared to the original lightweight baseline in order to increase feature extraction capacity and restoration quality.

---

## Training Strategy

### Optimization

* Optimizer: AdamW
* Learning rate scheduler: Cosine Annealing
* Mixed precision training (AMP)
* Exponential Moving Average (EMA)
* Automatic checkpoint saving

### Loss Functions

The training objective is mainly based on:

* L1 reconstruction loss

Validation performance is monitored using:

* PSNR

### Training Configuration

* Epochs: 80
* Patch size: 256 × 256
* Batch size adjusted according to GPU memory
* Training performed from scratch

The best model checkpoint is automatically selected according to validation PSNR.

---

## Usage

### Train the model

```bash
python train.py
```

### Run inference and generate pred.npz

```bash
python inference.py
```

The output file follows the competition format:

```python
{
    "0.png": np.ndarray(shape=(3, H, W)),
    "1.png": np.ndarray(shape=(3, H, W)),
    ...
}
```

The final submission file must be named:

```bash
pred.npz
```

---

## Performance Snapshot

### Final Validation Performance

* Best validation PSNR: ~27.64 dB

This score is significantly above the weak baseline defined in the homework instructions.

### Training Behavior

The training process showed stable convergence across epochs thanks to:

* EMA stabilization
* Cosine learning rate scheduling
* Mixed precision training

### Competition Performance

Final public leaderboard score:

* [INSERT YOUR FINAL CODABENCH SCORE HERE]

Leaderboard screenshot:

```md
[Insert leaderboard screenshot here]
```

### Qualitative Results

The model successfully removes:

* Rain streaks
* Snow artifacts
* Degradation noise

while preserving:

* Image structures
* Edges
* Global textures
* Scene geometry

Some difficult samples with extremely dense degradations may still produce slightly blurred outputs or residual artifacts.

---

## Additional Experiments

### Impact of EMA

EMA stabilization improved validation stability and produced slightly higher PSNR values during inference.

### Importance of Patch-Based Training

Patch extraction significantly reduced GPU memory usage and increased local texture diversity during training.

### Test-Time Augmentation (TTA)

TTA was used during inference in order to improve prediction robustness and stabilize restoration quality.

---

## Results

### Train / Val losses

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/b25b93e5-e4be-4a39-b222-ec902b5dbfec" />

### PSNR

<img width="687" height="470" alt="image" src="https://github.com/user-attachments/assets/f31e2d87-1a76-4160-9842-d13b1e73d1b2" />

### Results visualization

<img width="1470" height="594" alt="image" src="https://github.com/user-attachments/assets/c662198c-ee34-43a7-baf9-625ad55002a8" />

### Codabench leaderboard

<img width="890" height="37" alt="image" src="https://github.com/user-attachments/assets/e3d68842-5147-402c-aa63-66e39e887aee" />

---

## References

1. PromptIR: Prompting for All-in-One Blind Image Restoration
2. PyTorch Documentation
3. Visual Recognition using Deep Learning 2026 Spring - Homework 4 Slides
4. PromptIR GitHub Repository
