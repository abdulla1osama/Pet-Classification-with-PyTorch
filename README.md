# Pet-Classification-with-PyTorch
# 🐱🐶 Oxford-IIIT Pet Classification with ResNet18

A deep learning project for classifying 37 different pet breeds using transfer learning with PyTorch and ResNet18.

## 📊 Project Overview

This project demonstrates transfer learning for image classification on the Oxford-IIIT Pet Dataset. By fine-tuning a pretrained ResNet18 model, we achieve **~90% validation accuracy** on 37 pet breed categories.

### Key Features
- ✅ Transfer learning with ImageNet-pretrained ResNet18
- ✅ Data augmentation for improved generalization
- ✅ Clean, production-ready PyTorch code
- ✅ Model checkpointing and progress tracking
- ✅ Configurable training via command-line arguments

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~96-97% |
| **Validation Accuracy** | ~88-89% | |
| **Model Size** | 44.7 MB |

---

## 📁 Project Structure

```
pet-classification/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
├── classification.py                 # Main training script 
├── test.py                           #Test script                
│
├── best_model.pt                       # Best model checkpoint (auto-generated)
│
├── data/
│   └── oxford-iiit-pet/              # Dataset (auto-downloaded)
│       ├── images/
│       ├── annotations/
│       └── ...
│
```
#change the download parameter to True in classification file to download the dataset

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pet-classification.git
cd pet-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
# Basic training (Windows)
python train.py
