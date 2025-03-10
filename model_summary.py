import torch
from torchsummary import summary
import math

# Import the PyramidNet model and related functions (assuming they're in the same directory)
from model import create_pyramidnet, PyramidNet, BottleneckBlock, BasicBlock

def count_parameters(model):
    """Function to count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Check both bottleneck and basic block versions
configs = [
    # First with bottleneck blocks
    ([1, 1, 1], True, 270, "Bottleneck"),
    # Then with basic blocks
    ([1, 1, 1], False, 270, "Basic"),
    # Optional: Check an even smaller model
    ([1, 1, 1], False, 150, "Basic (smaller alpha)")
]

print("PyramidNet Model Parameter Summary:")
print("=" * 50)

for config in configs:
    blocks, use_bottleneck, alpha, name = config
    
    # Create model
    model = create_pyramidnet(blocks, use_bottleneck, alpha)
    
    # Count parameters
    params = count_parameters(model)
    
    print(f"\n{name} block configuration:")
    print(f"Blocks: {blocks}, Alpha: {alpha}")
    print(f"Total parameters: {params:,}")
    print(f"Meets 5M limit: {'Yes' if params < 5000000 else 'No'}")
    
    # Get more detailed summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print("\nModel structure:")
        # Move to device (add exception handling in case of memory issues)
        model = model.to(device)
        summary(model, (3, 32, 32))
    except Exception as e:
        print(f"Error during model summary: {e}")
        # If CUDA out of memory, try on CPU
        if "CUDA out of memory" in str(e):
            try:
                print("Retrying on CPU...")
                summary(model.cpu(), (3, 32, 32))
            except Exception as e2:
                print(f"CPU summary also failed: {e2}")
    
    print("=" * 50)

print("\nRecommendation:")
print("For a PyramidNet under 5M parameters, use the Basic block configuration with [1,1,1] layers and alpha=150.")