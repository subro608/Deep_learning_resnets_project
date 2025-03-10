import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
from model import create_pyramidnet
from dataset import CIFAR10Dataset
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
from model import create_pyramidnet
from dataset import CIFAR10Dataset
from PIL import ImageFilter, Image
import random

class RandomBlur:
    def __init__(self, p=0.5, blur_limit=(0.5, 1.5)):
        self.p = p
        self.blur_limit = blur_limit
    
    def __call__(self, img):
        if random.random() < self.p:
            blur_type = random.choice(['gaussian', 'box', 'motion'])
            blur_strength = random.uniform(self.blur_limit[0], self.blur_limit[1])
            
            if blur_type == 'gaussian':
                return img.filter(ImageFilter.GaussianBlur(radius=blur_strength))
            elif blur_type == 'box':
                return img.filter(ImageFilter.BoxBlur(radius=int(blur_strength)))
            elif blur_type == 'motion':
                # Approximate motion blur using directional box blur
                kernel_size = int(blur_strength * 3)
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Make sure kernel size is odd
                return img.filter(ImageFilter.GaussianBlur(radius=blur_strength))
        return img
def evaluate(model, data_loader, device, criterion=None):
    """Evaluate model on the provided data loader"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = running_loss / len(data_loader) if criterion is not None else None
    
    return accuracy, avg_loss
def plot_metrics(metrics, save_path='metrics_plots'):
    """Plot and save training and validation metrics"""
    os.makedirs(save_path, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Plot losses
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(save_path, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual plots for more detail
    # Loss plot and Accuracy plot code...
def train_model(num_blocks=[2, 2, 2], use_bottleneck=True, alpha=270):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up transforms with augmentation for best model yet
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    
    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    
    # Set up transforms with augmentation including blur
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # Convert to PIL for custom blur transform
        RandomBlur(p=0.7, blur_limit=(0.5, 2.0)),  # High probability of applying blur
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),  # Convert back to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # For test data, we can add a slight blur to match test conditions
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        # Optional: Add a fixed mild blur to validation/test to better match test conditions
        # lambda x: x.filter(ImageFilter.GaussianBlur(radius=0.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # Load datasets
    train_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_files = ["test_batch"]
    
    dataset = CIFAR10Dataset(train_files, transform=train_transform)
    test_dataset = CIFAR10Dataset(test_files, transform=test_transform)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))  # Using more data for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize PyramidNet model
    model_type = "bottleneck" if use_bottleneck else "basic"
    print(f"Creating PyramidNet model with {model_type} blocks, alpha={alpha}, layers={num_blocks}")
    
    model = create_pyramidnet(
        num_blocks, 
        use_bottleneck=use_bottleneck, 
        alpha=alpha, 
        num_classes=10
    )
    model.to(device)
    
    # Create directory for saved models
    os.makedirs('saved_models', exist_ok=True)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    
    # Using SGD with momentum and weight decay (better for ResNets)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.1,  # Starting with a higher learning rate
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_val_acc = 0.0
    early_stop_counter = 0
    max_early_stop = 80  # Stop if no improvement for 20 epochs
    
    for epoch in range(1000):  # More epochs for deeper training
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        
        # Validation
        val_acc = evaluate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint (every 10 epochs to save space)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'saved_models/pyramidnet_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            best_model_path = f'saved_models/best_pyramidnet.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= max_early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs with no improvement")
            break
    
    # Save final model
    final_model_path = f'saved_models/final_pyramidnet.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Test on test dataset
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy with best model: {test_acc:.4f}")
    
    return model

if __name__ == "__main__":
    train_model()