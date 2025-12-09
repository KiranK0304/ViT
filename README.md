📘 Vision Models — Vision Transformer (ViT) from Scratch in PyTorch

This repository contains a clean and modular implementation of the Vision Transformer (ViT) architecture built entirely from scratch using PyTorch.
It includes everything needed for training, evaluating, and running inference on image classification tasks such as CIFAR-10.

The project is structured to be simple, readable, and easy to extend.

🚀 Features
Vision Transformer (ViT) Implementation

Patch Embedding via Conv2D

Learnable Class Token

Learnable Positional Embedding

Multi-Head Self-Attention (MHSA)

MLP block with GELU activation

Pre-LayerNorm structure

Residual connections

Training Pipeline

CIFAR-10 loading and preprocessing

Standard augmentations

Configurable hyperparameters

Training loop with loss & accuracy tracking

Evaluation Tools

Test accuracy calculation

Confusion matrix

Per-class accuracy visualization

Inference

Script for running predictions on custom images

🧩 Repository Structure
vision-models/
│
├── model.py          # Vision Transformer architecture
├── dataset.py        # Dataset loading + transforms
├── train.py          # Training loop
├── evaluate.py       # Evaluation utilities
├── inference.py      # Inference on new images
├── config.py         # Hyperparameters and settings
└── README.md

📦 Installation
git clone https://github.com/KiranK0304/vision-models.git
cd vision-models
pip install -r requirements.txt   # if available

🏋️‍♂️ Training

Train the ViT model on CIFAR-10:

python train.py


Training progress (loss & accuracy) will be printed in the terminal.

🔍 Inference on Custom Images
python inference.py --image path/to/image.jpg --checkpoint vit_cifar10.pth


This will output the predicted class.

📈 Evaluation Outputs

After training, the repository can generate:

Test Accuracy

Confusion Matrix Plot

Per-Class Accuracy Chart

These help visualize how the model performs across different categories.

📜 License

This project is intended for educational and research use.
Feel free to fork, modify, and experiment.
