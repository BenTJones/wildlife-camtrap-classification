# Wildlife Image Classification (CCT20)

This project creates and uses an image classification pipeline on a subset of the Caltech Camera Traps (CCT20) dataset.  
A convolutional neural network (EfficientNet-B0) was trained in PyTorch to classify animal species from camera trap images.
The goal was to build and train a functional deep learning model on real-world imagery using limited compute resources, demonstrating the full work flow necessary from data preparation to model evaluation.

## Objective

Develop and evaluate an end-to-end image classification model capable of identifying wildlife species from camera trap images.  
The focus was on understanding the process and being able to set up image deep learning pipelines rather than achieving maximum accuracy.

## Technical Overview

| **Component** | **Details** |
| Dataset | Caltech Camera Traps (CCT20) – 260K images (subset used: ~20%) |
| Model | EfficientNet-B0 (pretrained on ImageNet) |
| Framework | PyTorch |
| Image Size | 224 × 224 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Split | 80/10/10 (StratifiedGroupKFold by location and label) |
| Hardware | CPU-only training |
| Epochs | 6 |
| Learning Rate | 1e-3 |

## Results Summary

| Metric | Value |
| Validation Accuracy | ~61.6% |
| Validation Loss | 0.043 |
| Best Epoch | 6 |

Training and validation metrics indicate stable convergence with mild overfitting, expected given dataset imbalance and limited compute.

### Key Performance Figures

**Accuracy and Loss Curves**  
![Train vs Val Accuracy](reports/figures/Train_Val_Acc.png)  
![Train vs Val Loss](reports/figures/Train_Val_Loss.png)

**Normalized Confusion Matrix**  
![Confusion Matrix](reports/figures/Confusion_matrix.png)

**Per-Class F1 Scores**  
![Per-Class F1](reports/figures/Class_f1.png)

## Repository Structure

