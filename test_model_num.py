import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def test_digit_classifier(image_path, model_path):
    # 1. Load the trained classification model
    model = YOLO(model_path)

    # 2. Read the input image
    img = cv2.imread(str(image_path))
    if img is None:
        print("Error: Could not read image.")
        return

    # 3. Preprocess: Ensure it is 100x100 (matching your training imgsz)
    img_resized = cv2.resize(img, (100, 100))
    
    # 4. Perform Inference
    results = model(img_resized)
    result = results[0]

    # 5. Extract results
    probs = result.probs  # Confidence scores for all classes
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    label = result.names[top1_idx]

    # 6. Visualization
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {label} ({top1_conf:.2%})")
    plt.axis("off")
    
    # Print the full list of probabilities for debugging
    print(f"\n--- Results for {Path(image_path).name} ---")
    for i, conf in enumerate(probs.data.tolist()):
        print(f"Class {result.names[i]}: {conf:.4f}")
    
    plt.show()

if __name__ == "__main__":
    # Update these paths to your actual files
    MY_IMAGE = "test/right_number_crop.jpg" 
    MY_MODEL = "models/best_num_classify.pt"
    
    test_digit_classifier(MY_IMAGE, MY_MODEL)