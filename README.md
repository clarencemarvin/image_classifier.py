# Image Classifier CNN

This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch. The CNN is designed to classify images into 10 different classes.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib

## Dataset

The code assumes that your dataset is organized in the following structure:
data/
train/
class_0/
class_1/
...
class_9/
test/
class_0/
class_1/
...
class_9/

Each class folder should contain the respective images.

## Features

- Implements a CNN with convolutional layers, max pooling, and dropout
- Uses data augmentation for the training set
- Implements train, validation, and test splits
- Uses Adam optimizer with OneCycleLR learning rate scheduler
- Provides detailed training and validation metrics for each epoch
- Calculates and displays class-wise accuracies on the test set

## Usage

1. Ensure your dataset is in the correct format and location.
2. Run the script: python classifier.py
3. The script will train the model and display training progress, final accuracies, and other metrics.

## Model Architecture

The CNN architecture is defined in the `Net` class and consists of:
- 4 Convolutional layers
- 2 Max pooling layers
- Dropout layers
- 2 Fully connected layers

## Results
