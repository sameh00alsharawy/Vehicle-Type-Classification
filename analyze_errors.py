import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
from tqdm import tqdm

# --- 1. Model Loading ---
def load_trained_model(model_path, num_classes):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() 
    return model

# --- 2. Batch Explainability Logic ---
def generate_error_heatmaps(csv_path, data_dir, model, class_names):
    output_dir = "error_heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(320), 
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_layer = model.features[-1]
    grad_cam = LayerGradCam(model, target_layer)
    
    # Read the CSV
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        
    print(f"Found {len(rows)} misclassified images. Generating heatmaps...")
    
    for row in tqdm(rows, desc="Processing Errors"):
        file_name = row['File Name']
        true_class = row['True Class']
        pred_class = row['Predicted Class']
        
        # Construct the exact path to the original image
        image_path = os.path.join(data_dir, 'Testing', true_class, file_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Could not find {image_path}. Skipping.")
            continue
            
        # Process Image
        raw_img = Image.open(image_path).convert('RGB')
        input_tensor = transform(raw_img).unsqueeze(0)
        
        # We target the class the model *falsely predicted* to see why it was confused
        pred_idx = class_names.index(pred_class)
        attributions = grad_cam.attribute(input_tensor, target=pred_idx)
        attributions_resized = LayerGradCam.interpolate(attributions, (300, 300))
        
        # Format for visualization
        img_viz = input_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
        img_viz = img_viz * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_viz = np.clip(img_viz, 0, 1)
        attr_viz = attributions_resized.squeeze().detach().numpy()
        
        # Generate the visual without showing the window
        fig, axis = viz.visualize_image_attr_multiple(
            attr_viz,
            img_viz,
            methods=["original_image", "blended_heat_map"],
            signs=["all", "positive"],
            titles=[f"True: {true_class}", f"Grad-CAM (Guessed: {pred_class})"],
            show_colorbar=True,
            fig_size=(10, 5),
            use_pyplot=False # Prevent it from opening a GUI window
        )
        
        # Save and close to prevent memory leaks
        save_name = f"{true_class}_as_{pred_class}_{file_name.split('.')[0]}.png"
        fig.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    # UPDATE THESE PATHS
    DATA_DIR = r"C:\Users\sameh\Desktop\XAI\vehicle" 
    MODEL_WEIGHTS = "best_interpretable_model.pth"
    CSV_PATH = "misclassified_report.csv"
    
    # Must match the exact order of your training dataset classes
    CLASS_NAMES = ['Auto Rickshaws', 'Bikes', 'Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains']
    
    print("Loading model...")
    model = load_trained_model(MODEL_WEIGHTS, len(CLASS_NAMES))
    
    generate_error_heatmaps(CSV_PATH, DATA_DIR, model, CLASS_NAMES)
    print("\n--> Done! Check the 'error_heatmaps' folder.")