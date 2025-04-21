import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import Optional

try:
    from .model import get_model
except ImportError:
    from model import get_model

try:
    from .utils import display_prediction
except ImportError:
    from utils import display_prediction


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    model_weights_path: str,
    device: torch.device,
    img_size: int = 224
) -> Optional[str]:
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at '{model_weights_path}'")
        return None

    try:
        state_dict = torch.load(model_weights_path, map_location=device)

        if list(state_dict.keys())[0].startswith('module.'):
             state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model weights loaded successfully from {model_weights_path}")

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device)
        print(f"Image loaded and preprocessed: {os.path.basename(image_path)}")

        prediction_label = "Error during prediction"
        with torch.no_grad():
            output = model(img_tensor)
            probability = torch.sigmoid(output).item()

            if probability > 0.5:
                prediction_label = f"Tumor Detected (Prob: {probability:.4f})"
            else:
                prediction_label = f"No Tumor (Prob: {probability:.4f})"

        print(f"Prediction complete: {prediction_label}")

        display_prediction(image_path, prediction_label)

        return prediction_label.split(' (')[0]

    except FileNotFoundError as fnf_error:
         print(f"Error: File not found during prediction. {fnf_error}")
         return None
    except Exception as e:
        print(f"An error occurred during prediction for '{image_path}': {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Running predictor example...")

    MODEL_WEIGHTS = "../models/brain_tumor_model_epoch_10.pth"
    TEST_IMAGE = "../archive/test/Y250.jpg"
    DEVICE_STR = "mps" # cuda, cpu, mps

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Example model weights '{MODEL_WEIGHTS}' not found. Cannot run example.")
        exit()
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Example test image '{TEST_IMAGE}' not found. Cannot run example.")
        exit()

    if DEVICE_STR == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if DEVICE_STR == "mps":
            print("Warning: MPS not available, falling back to CPU.")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    try:
        model_instance = get_model(device=device)
    except Exception as model_init_error:
        print(f"Error initializing model: {model_init_error}")
        print("Ensure the BrainTumorModel class is correctly defined and imported.")
        exit()

    predicted_label = predict_image(
        model=model_instance,
        image_path=TEST_IMAGE,
        model_weights_path=MODEL_WEIGHTS,
        device=device
    )

    if predicted_label:
        print(f"\nExample prediction finished. Result: {predicted_label}")
    else:
        print("\nExample prediction failed.")
