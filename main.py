import argparse
import torch
from torchsummary import summary

from model import create_pyramidnet
from train import train_model
from inference import run_inference, load_model

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 PyramidNet Training and Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'both'],
                        help='Operation mode: train, inference, or both')
    parser.add_argument('--test_data', type=str, default='cifar_test_nolabel.pkl',
                        help='Path to test data file')
    parser.add_argument('--show_model', action='store_true',
                        help='Show model summary')
    parser.add_argument('--bottleneck', type=bool, default=True,
                        help='Use bottleneck blocks in PyramidNet')
    parser.add_argument('--alpha', type=int, default=270,
                        help='Alpha parameter for PyramidNet (widening factor)')
    parser.add_argument('--blocks', type=int, nargs=3, default=[4, 4, 4],
                        help='Number of blocks in each layer of PyramidNet')
    
    args = parser.parse_args()
    
    if args.show_model:
        print("Model Summary:")
        model = create_pyramidnet(args.blocks, use_bottleneck=args.bottleneck, alpha=args.alpha)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        summary(model, (3, 32, 32))
    
    if args.mode == 'train' or args.mode == 'both':
        print("Starting training...")
        model = train_model(args.blocks, use_bottleneck=args.bottleneck, alpha=args.alpha)
        
        if args.mode == 'both':
            print("\nStarting inference with trained model...")
            run_inference(model, args.test_data)
    
    elif args.mode == 'inference':
        print("Loading model for inference...")
        model = load_model(args.blocks, use_bottleneck=args.bottleneck, alpha=args.alpha)
        run_inference(model, args.test_data)

if __name__ == "__main__":
    main()