# PyramidNet for CIFAR-10 Classification

This repository contains the implementation of PyramidNet, a deep residual network architecture that gradually increases feature dimensions throughout the network, applied to the CIFAR-10 image classification task. Our implementation achieves strong classification performance while maintaining parameter efficiency (under 5M parameters).

## Repository Overview

The main components of this repository include:

### Comprehensive Notebook
- **Deep_residuals.ipynb**: A complete end-to-end notebook containing all aspects of the project from data preprocessing to training, inference, and model architecture details. This is the best place to start if you want to understand the full pipeline.

### Individual Python Scripts
If you prefer working with separate Python scripts, the repository includes:

- **dataset.py**: Implements data loading and preprocessing for the CIFAR-10 dataset, including augmentation
- **model.py**: Contains PyramidNet architecture implementation with various block types (Basic and Bottleneck)
- **train.py**: Script for training the model with various hyperparameters and learning rate schedules
- **test.py**: Evaluates the trained model on the test set and generates predictions
- **inference.py**: Provides utilities for making predictions on new images
- **model_summary.py**: Generates parameter counts and model architecture summaries
- **classwise_dataset_dist.py**: Analyzes the class distribution in the CIFAR-10 dataset

### Results and Visualizations
- **confusion_matrix.png**: Visualization of model performance across different classes
- **training_metrics.png**: Plots of training/validation loss and accuracy
- **misclassified_examples.png**: Examples of images that the model failed to classify correctly
- **classification_report.txt**: Detailed metrics on model performance
- **kaggle_submission.csv**: Predictions formatted for Kaggle competition submission

## Model Architecture

The implemented PyramidNet architecture features:
- Gradual feature dimension growth controlled by the alpha parameter
- Modified bottleneck blocks without the traditional expansion factor
- Three stages of residual blocks with downsampling between stages
- Global average pooling followed by a fully connected classification layer

## Performance

Our final PyramidNet model (Bottleneck, [4,4,4], α=270) achieves 83.50%/83.66% accuracy on the CIFAR-10 Kaggle test set while staying under 5M parameters.

## Usage

To get started, either:
1. Run the comprehensive notebook `Deep_residuals.ipynb`
2. Use the individual Python scripts for specific tasks

### Training

To train the model from scratch, run:

```bash
python train.py
```

Alternatively, you can use the following command to train and visualize the model architecture:

```bash
python main.py --show_model --blocks 4 4 4 --alpha 270
```

### Evaluation

For evaluating a trained model:

```bash
python test.py
```

This setup allows for flexible model training and evaluation while ensuring optimal performance for CIFAR-10 classification.
