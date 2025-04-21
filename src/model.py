import torch
import torch.nn as nn
import torchvision.models as models

def get_model(device='cpu'):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model(device)
    print("\nModel Architecture:")
    print(model)

    example_input = torch.randn(1, 3, 224, 224).to(device)
    print(f"\nExample input shape: {example_input.shape}")

    try:
        with torch.no_grad():
            output = model(example_input)
        print(f"Example output shape: {output.shape}")
        print(f"Example output (logits): {output.item()}")
    except Exception as e:
        print(f"Error during forward pass: {e}")