# Final Report: Image Classification Project

## Abstract
This project implements an end-to-end image classification pipeline using a dataset obtained online.  
A convolutional neural network (CNN) architecture was trained and evaluated on preprocessed image data, efficientnetb0 from timm was selected for this task.  
The aim was to gain practical experience in computer vision workflows, not to achieve peak accuracy, due to limitations of my PC.

## 1. Introduction
The objective of this project was to explore the process of building an image classifier from scratch - from dataset preparation through model evaluation - using PyTorch and other relevant python libraries.  
The personal goal of this project was to develope my deep learning and ML model creation skills and learn to build and full pipeline for image classifaction that could be replicated in future and also help me develop my skills further.

## 2. Dataset
- **Source:** [CCT20 Dataset](https://lila.science/datasets/caltech-camera-traps)
- **Classes:** 22 in total only 16 trained on model due to low frequencies for some  
- **Preprocessing:**  Resized to 224x224, normalised to IMAGENET mean and SD and included random horizontal flipping of images
- **Split:** 80% Train, 10% Validation, 10% Test  
- **Notes:** As previously mentioned some classes had frequencys of 2 in the ~260k images so they had to be dropped especially when I had to reduce to ~40k images for my machine 


## 3. Methods
- **Model:** EfficientNet-B0
- **Transfer Learning:** Used in model  
- **Loss Function:** Cross Entropy Loss
- **Optimizer:** Adam
- **Epochs:** 10 with a patience of 3
- **Batch Size:** 32 
- **Learning Rate:** 0.001
- **Device:** CPU, implemnted with code to run on GPU but lack that in my PC

---

## 4. Results
**Training Results:**
See [Perforance Log.csv](performance_log.csv)

**Training Curves:**  
![Train vs Val Accuracy](figures/Train_Val_Acc.png)  
![Train vs Val Loss](figures/Train_Val_Loss.png)

**Confusion Matrix:**  
![Normalised Confusion Matrix](figures/Confusion_matrix.png)

**Per Class F1 Scores**
![Class F1 scores](figures/Class_f1.png)

**Classification Report:**  
See [classification_report.csv](classification_report.csv)

---

## 5. Discussion
- The model achieved 61% validation accuracy after 7 epochs.  
- The training curve indicate decreases in both test accuracy and loss over the epochs steadily and conitually decreasing. 
- The validation curve shows slight fluctuation with no massively unexpected trends but the low number of epochs make it difficult to find a pattern.
- The confusion matrix shows strongest performance on empty images and detecting them and weakness on skunk.  
- Per Class f1 scores show some classes 0,7,16 (empty,bobcat,lizard) have strong and confident predicition but the lower f1 scores likely correspond to the rarer classes with insufficient exposure to them leads to lack of pattern detection. The spread of scores highlight imbalances across the dataset.
- Limitations include the low frequencies of certain classes leading to some being purged and others. The machine itself limited the performance of the model as I couldnt train on the full dataset limiting exposure to certain classes and sometimes increasing the imbalance of class sizes.
- Overall the model showed capability to learn from the provided data and further improvements would include class rebalancing, trying alternative data augmentation and also training on GPU hardware to allow for the training using full dataset and more epochs.

## 6. Conclusion
This project successfully implemented and evaluated an image classification pipeline from a web-sourced dataset.  
Even with limited resources, the model learned meaningful representations of visual features.  
Future work could involve training with more data, additional architectures, or real-time inference deployment.


**Author:** Ben Jones  
**Date:** October 2025  
**Institution:** IMperial College London
