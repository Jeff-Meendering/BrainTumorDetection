import cv2
import os
import numpy as np
from PIL import Image

def display_prediction(image_path: str, prediction_label: str):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if "Tumor Detected" in prediction_label:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        position = (10, 30)

        cv2.putText(img_cv2, prediction_label, position, font, font_scale, color, thickness, cv2.LINE_AA)

        window_name = f"Prediction: {os.path.basename(image_path)}"
        cv2.imshow(window_name, img_cv2)
        print(f"Displaying prediction for {os.path.basename(image_path)}. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    except Exception as e:
        print(f"Error displaying image '{image_path}': {e}")

if __name__ == '__main__':
    test_image = "../archive/test/Y250.jpg"
    if os.path.exists(test_image):
         print(f"\nTesting display_prediction with real image: {test_image}")
         display_prediction(test_image, "Tumor Detected (Example)")
         display_prediction(test_image, "No Tumor (Example)")
    else:
        print(f"\nTest image '{test_image}' not found. Skipping real image test.")