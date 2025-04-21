import argparse
import torch
import os
from src.data_loader import get_dataloaders
from src.model import get_model
from src.trainer import train_model
from src.predictor import predict_image

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    parser = argparse.ArgumentParser(description="Brain Tumor Detection CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands: train, predict')
    subparsers.required = True

    parser_train = subparsers.add_parser('train', help='Train the brain tumor detection model')
    parser_train.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser_train.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser_train.add_argument('--data_dir', type=str, default='archive', help='Directory containing the dataset (default: archive)')
    parser_train.add_argument('--model_save_path', type=str, default='models/brain_tumor_model.pth', help='Path to save the trained model (default: models/brain_tumor_model.pth)')

    parser_predict = subparsers.add_parser('predict', help='Predict brain tumor from an image')
    parser_predict.add_argument('--image_path', type=str, required=True, help='Path to the input image for prediction')
    parser_predict.add_argument('--model_load_path', type=str, default='models/brain_tumor_model.pth', help='Path to load the trained model (default: models/brain_tumor_model.pth)')

    args = parser.parse_args()

    model = get_model(device=device)

    if args.command == 'train':
        print(f"Starting training for {args.epochs} epochs...")
        model_dir = os.path.dirname(args.model_save_path)
        if model_dir:
             os.makedirs(model_dir, exist_ok=True)
             print(f"Ensured directory exists: {model_dir}")

        print(f"Loading data from: {args.data_dir}")
        train_loader, val_loader = get_dataloaders(args.data_dir)

        if train_loader is None or val_loader is None:
            print("Error: Failed to create data loaders. Exiting training.")
            return

        print("Data loaders created successfully.")

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_path=args.model_save_path
        )
        print(f"Training complete. Model saved to {args.model_save_path}")

    elif args.command == 'predict':
        print(f"Starting prediction for image: {args.image_path}")
        prediction_label = predict_image(
            model=model,
            image_path=args.image_path,
            model_weights_path=args.model_load_path,
            device=device
        )
        if prediction_label:
            print(f"main.py received prediction: {prediction_label}")
        else:
            print("main.py: Prediction failed.")

if __name__ == "__main__":
    main()