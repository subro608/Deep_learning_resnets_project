import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataset import CIFAR10Dataset
import os

def analyze_cifar10_distribution():
    # Define the class names for CIFAR-10
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Load the training dataset
    train_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_files = ["test_batch"]
    
    # Check if files exist and print available files in directory
    print("Files in current directory:")
    for file in os.listdir('.'):
        if file.startswith("data_batch") or file == "test_batch":
            print(f"  - {file}")
    
    # Load the datasets
    print("\nLoading training data...")
    train_dataset = CIFAR10Dataset(train_files)
    
    print("Loading test data...")
    test_dataset = CIFAR10Dataset(test_files)
    
    # Count instances of each class in training set
    train_class_counts = np.zeros(10, dtype=int)
    for label in train_dataset.labels:
        train_class_counts[label] += 1
    
    # Count instances of each class in test set
    test_class_counts = np.zeros(10, dtype=int)
    for label in test_dataset.labels:
        test_class_counts[label] += 1
    
    # Print the distribution
    print("\nClass Distribution in Training Set:")
    print("================================")
    for i, (name, count) in enumerate(zip(class_names, train_class_counts)):
        percentage = (count / len(train_dataset.labels)) * 100
        print(f"Class {i} ({name}): {count} images ({percentage:.2f}%)")
    
    print("\nClass Distribution in Test Set:")
    print("============================")
    for i, (name, count) in enumerate(zip(class_names, test_class_counts)):
        percentage = (count / len(test_dataset.labels)) * 100
        print(f"Class {i} ({name}): {count} images ({percentage:.2f}%)")
    
    # Visualize the distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set distribution
    ax1.bar(class_names, train_class_counts, color='skyblue')
    ax1.set_title('Class Distribution in Training Set')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    
    # Test set distribution
    ax2.bar(class_names, test_class_counts, color='lightgreen')
    ax2.set_title('Class Distribution in Test Set')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cifar10_class_distribution.png')
    print("\nDistribution plot saved as 'cifar10_class_distribution.png'")
    
    # Display some sample images
    print("\nShowing sample images from each class...")
    plt.figure(figsize=(12, 10))
    
    for i in range(10):
        # Find the first image of this class
        idx = train_dataset.labels.index(i)
        img = train_dataset.data[idx]
        
        # Plot
        plt.subplot(2, 5, i+1)
        plt.imshow(np.transpose(img, (1, 2, 0)))  # Convert back to (H, W, C) for display
        plt.title(class_names[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_sample_images.png')
    print("Sample images saved as 'cifar10_sample_images.png'")
    
    # Additional analysis: check for class imbalance
    train_min = np.min(train_class_counts)
    train_max = np.max(train_class_counts)
    train_imbalance_ratio = train_max / train_min
    
    print(f"\nClass Imbalance Analysis:")
    print(f"Training set min class count: {train_min}")
    print(f"Training set max class count: {train_max}")
    print(f"Imbalance ratio: {train_imbalance_ratio:.2f}")
    
    if train_imbalance_ratio > 1.1:
        print("Note: There's some class imbalance in the dataset.")
        print("Consider using class weights or data augmentation techniques during training.")
    else:
        print("The dataset is well-balanced across all classes.")

if __name__ == "__main__":
    analyze_cifar10_distribution()