import torch
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from model import create_pyramidnet

# def load_model(num_blocks=[3, 3, 3], use_bottleneck=True, alpha=270):
#     model_path = 'saved_models/best_pyramidnet.pth'
#     model = create_pyramidnet(num_blocks, use_bottleneck=use_bottleneck, alpha=alpha, num_classes=10)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model
def load_model(num_blocks=[3, 3, 3], use_bottleneck=True, alpha=270):
    model_path = 'saved_models/best_pyramidnet.pth'
    checkpoint = torch.load(model_path)
    model = create_pyramidnet(num_blocks, use_bottleneck=use_bottleneck, alpha=alpha, num_classes=10)
    
    # Extract the model state dictionary from the checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # For backward compatibility with older saved models
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model
def run_inference(model, test_data_path="cifar_test_nolabel.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data directly from pickle file
    print("Loading test data...")
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
    
    # Extract images and IDs
    test_images = torch.tensor(test_data[b"data"], dtype=torch.float32) / 255.0  # Normalize
    ids = test_data[b'ids'] if b'ids' in test_data else np.arange(len(test_data[b"data"]))  # Extract IDs or create sequential IDs
    
    # Reshape images to (N, 3, 32, 32) if needed
    if len(test_images.shape) == 2:  # If flattened (N, 3072)
        test_images = test_images.reshape(-1, 3, 32, 32)
    elif test_images.shape[-1] == 3:  # If (N, 32, 32, 3)
        test_images = test_images.permute(0, 3, 1, 2)  # Convert to (N, 3, 32, 32)
    
    # Apply mean/std normalization (CIFAR-10 standard)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    test_images = (test_images - mean) / std
    
    # Create Dataset and DataLoader
    batch_size = 128
    test_dataset = TensorDataset(test_images)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Move model to device
    model.to(device)
    
    # Run inference in batches
    predictions = []
    print("Running inference...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch[0].to(device)  # Move batch to GPU
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)  # Get predicted class
            predictions.extend(preds.cpu().numpy())  # Move to CPU and store
    
    # Create submission file
    print("Creating submission file...")
    # Ensure ids and predictions are 1D arrays
    ids = np.array(ids).flatten()
    predictions = np.array(predictions).flatten()
    
    # Stack IDs and predictions together
    submission = np.column_stack((ids, predictions))
    
    # Save as CSV with proper column headers
    np.savetxt("kaggle_submission.csv", submission, delimiter=",", header="ID,Labels", fmt="%d", comments="")
    print("✅ Submission file saved as kaggle_submission.csv")

if __name__ == "__main__":
    # This allows running inference directly from this script
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on CIFAR-10 test data')
    parser.add_argument('--test_data', type=str, default='cifar_test_nolabel.pkl',
                      help='Path to test data file')
    parser.add_argument('--bottleneck', type=bool, default=True,
                      help='Use bottleneck blocks in PyramidNet')
    parser.add_argument('--alpha', type=int, default=270,
                      help='Alpha parameter for PyramidNet (widening factor)')
    parser.add_argument('--blocks', type=int, nargs=3, default=[4, 4, 4],
                      help='Number of blocks in each layer of PyramidNet')
    
    args = parser.parse_args()
    
    print(f"Loading PyramidNet model with {'bottleneck' if args.bottleneck else 'basic'} blocks, alpha={args.alpha}, blocks={args.blocks}")
    model = load_model(args.blocks, args.bottleneck, args.alpha)
    run_inference(model, args.test_data)
