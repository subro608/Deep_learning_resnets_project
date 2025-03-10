import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import create_pyramidnet
from dataset import CIFAR10Dataset
import seaborn as sns

def evaluate_on_test_batch(num_blocks=[1, 1, 1], use_bottleneck=False, alpha=150):
    """
    Evaluate the trained PyramidNet model on the standard CIFAR-10 test batch
    and generate detailed performance metrics.
    """
    # Define the class names for CIFAR-10
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up transforms (same as used during training)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load the test dataset (standard CIFAR-10 test batch)
    test_file = ["test_batch"]
    test_dataset = CIFAR10Dataset(test_file, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load the model with the same configuration used during training
    model = create_pyramidnet(num_blocks, use_bottleneck=use_bottleneck, alpha=alpha, num_classes=10)
    
    # Print model configuration
    block_type = "Bottleneck" if use_bottleneck else "Basic"
    print(f"PyramidNet configuration: {block_type} blocks, alpha={alpha}, layers={num_blocks}")
    
    # Calculate parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {param_count:,}")
    
    # Load the trained model weights
    try:
        model_path = 'saved_models/best_pyramidnet.pth'
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've trained the model with this configuration first.")
        return
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Evaluate the model
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # For class-wise accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
            
            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    # Calculate class-wise accuracy
    print("\nClass-wise Accuracy:")
    print("===================")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Class {i} ({class_names[i]}): {accuracy:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Print classification report
    print("\nClassification Report:")
    print("=====================")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Save results to file
    with open("test_evaluation_results.txt", "w") as f:
        f.write(f"PyramidNet Model Evaluation on CIFAR-10 Test Set\n")
        f.write(f"Configuration: {block_type} blocks, alpha={alpha}, layers={num_blocks}\n")
        f.write(f"Parameter count: {param_count:,}\n\n")
        f.write(f"Overall Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Class-wise Accuracy:\n")
        for i in range(10):
            class_acc = 100 * class_correct[i] / class_total[i]
            f.write(f"Class {i} ({class_names[i]}): {class_acc:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    
    print("\nDetailed evaluation results saved to 'test_evaluation_results.txt'")
    
    # Find and analyze errors
    print("\nAnalyzing misclassified examples...")
    
    # Run one more time to get some misclassified examples
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified examples
            mask = (predicted != labels)
            if torch.any(mask):
                misclassified_idx = torch.where(mask)[0]
                misclassified_images.extend(images[misclassified_idx].cpu().numpy())
                misclassified_labels.extend(labels[misclassified_idx].cpu().numpy())
                misclassified_preds.extend(predicted[misclassified_idx].cpu().numpy())
                
                # Only collect up to 20 examples
                if len(misclassified_images) >= 20:
                    break
    
    # Plot some misclassified examples
    num_to_show = min(10, len(misclassified_images))
    if num_to_show > 0:
        plt.figure(figsize=(15, 8))
        for i in range(num_to_show):
            plt.subplot(2, 5, i+1)
            
            # Convert the normalized image back for display
            img = misclassified_images[i].transpose(1, 2, 0)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            plt.title(f"True: {class_names[misclassified_labels[i]]}\nPred: {class_names[misclassified_preds[i]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_examples.png')
        print("Misclassified examples saved as 'misclassified_examples.png'")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate PyramidNet on CIFAR-10 test batch')
    parser.add_argument('--bottleneck', type=bool, default=False,
                      help='Use bottleneck blocks in PyramidNet')
    parser.add_argument('--alpha', type=int, default=270,
                      help='Alpha parameter for PyramidNet (widening factor)')
    parser.add_argument('--blocks', type=int, nargs=3, default=[1, 1, 1],
                      help='Number of blocks in each layer of PyramidNet')
    
    args = parser.parse_args()
    
    evaluate_on_test_batch(args.blocks, args.bottleneck, args.alpha)