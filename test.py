import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import create_pyramidnet
from dataset import CIFAR10Dataset

def load_model(model_path, num_blocks=[4, 4, 4], use_bottleneck=True, alpha=270, num_classes=10):
    """Load a trained PyramidNet model from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture with the correct parameters
    model_type = "bottleneck" if use_bottleneck else "basic"
    print(f"Loading PyramidNet with {model_type} blocks, alpha={alpha}, layers={num_blocks}")
    
    model = create_pyramidnet(
        num_blocks, 
        use_bottleneck=use_bottleneck, 
        alpha=alpha, 
        num_classes=num_classes
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def evaluate(model, data_loader, device, verbose=True):
    """Evaluate model on the provided data loader"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = running_loss / len(data_loader)
    
    if verbose:
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy, avg_loss, all_predictions, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix for model predictions"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm

def analyze_results(y_true, y_pred, class_names, save_dir='test_results'):
    """Analyze model predictions and generate reports"""
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    
    # Generate and save classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Generate and save confusion matrix
    cm = plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_misclassified_examples(model, data_loader, device, class_names, num_examples=10, save_dir='test_results'):
    """Plot examples of images that were misclassified by the model"""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                misclassified_images.append(images[idx].cpu())
                misclassified_labels.append(labels[idx].item())
                misclassified_preds.append(preds[idx].item())
                
                if len(misclassified_images) >= num_examples:
                    break
            
            if len(misclassified_images) >= num_examples:
                break
    
    # If no misclassified examples found, return
    if not misclassified_images:
        print("No misclassified examples found.")
        return
    
    # Plot the misclassified examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if num_examples >= 10 else plt.subplots(1, num_examples, figsize=(15, 3))
    axes = axes.flatten()
    
    # CIFAR-10 mean and std for denormalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    for i, (img, true_label, pred_label) in enumerate(zip(misclassified_images, misclassified_labels, misclassified_preds)):
        if i >= len(axes):
            break
            
        # Denormalize image
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        
        # Plot image
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'misclassified_examples.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test PyramidNet on CIFAR-10 test set')
    parser.add_argument('--model_path', type=str, default='saved_models/best_pyramidnet.pth',
                        help='Path to the saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='Test batch size')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 4, 4],
                       help='Number of blocks in each pyramid stage')
    parser.add_argument('--use_bottleneck', action='store_true', default=True,
                       help='Use bottleneck blocks (default: True)')
    parser.add_argument('--alpha', type=float, default=270, help='PyramidNet alpha parameter')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(
        args.model_path, 
        num_blocks=args.num_blocks, 
        use_bottleneck=args.use_bottleneck, 
        alpha=args.alpha
    )
    
    # Print model info
    epoch = checkpoint.get('epoch', 'N/A')
    val_acc = checkpoint.get('val_acc', 'N/A')
    print(f"Loaded model from epoch: {epoch}")
    print(f"Validation accuracy: {val_acc}")
    
    # Load test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_files = ["test_batch"]
    test_dataset = CIFAR10Dataset(test_files, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    accuracy, loss, all_predictions, all_labels = evaluate(model, test_loader, device)
    
    # Analyze results
    print("\nAnalyzing results...")
    metrics = analyze_results(all_labels, all_predictions, class_names)
    
    # Plot misclassified examples
    print("\nPlotting misclassified examples...")
    plot_misclassified_examples(model, test_loader, device, class_names)
    
    print(f"\nTest completed successfully. Results saved to 'test_results' directory.")

if __name__ == "__main__":
    main()
