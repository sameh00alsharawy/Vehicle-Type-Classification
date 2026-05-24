import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

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

# --- 2. Inference & Confidence Logic ---
def analyze_image(image_path, model, class_names, apply_grayscale=False):
    """Runs an image through the model and returns the prediction and confidence."""
    
    # 1. Build the dynamic transform pipeline
    transform_list = [
        transforms.Resize(320), 
        transforms.CenterCrop(300)
    ]
    
    if apply_grayscale:
        # EfficientNet strictly requires 3 color channels. 
        # This converts the image to grayscale, but copies that gray across R, G, and B.
        transform_list.append(transforms.Grayscale(num_output_channels=3))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose(transform_list)
    
    # 2. Load and process
    try:
        raw_img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return None, None
        
    input_tensor = transform(raw_img).unsqueeze(0)
    
    # Move to GPU if available
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 3. Predict and calculate confidence
    with torch.no_grad():
        # Raw logits output
        outputs = model(input_tensor)
        
        # Softmax converts raw logits into percentages that add up to 100%
        probabilities = F.softmax(outputs, dim=1)[0]
        
    # Get the highest percentage and its corresponding class index
    top_prob, top_class_idx = torch.max(probabilities, 0)
    pred_class = class_names[top_class_idx]
    
    return pred_class, top_prob.item() * 100

# --- 3. Execution Block ---
if __name__ == "__main__":
    MODEL_WEIGHTS = "best_interpretable_model.pth"
    CLASS_NAMES = ['Auto Rickshaws', 'Bikes', 'Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains']
    
    print("Loading model for adversarial testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(MODEL_WEIGHTS, len(CLASS_NAMES)).to(device)
    
    # --- YOUR HYPOTHESIS TEST CASES ---
    # Put some test images in your project folder and update these names
    tests = [
        {"name": "1. Just Sky01", "path": "test_sky.jpg", "grayscale": False},
        {"name": "2. Just Sky02", "path": "test_sky02.jpg", "grayscale": False},
        {"name": "2. Just Sea", "path": "test_sea.jpg", "grayscale": False},
        {"name": "3. Train Tracks (No Train)", "path": "test_tracks.jpg", "grayscale": False},
        {"name": "4. Vertical Fences (Not Tracks)", "path": "test_fences.jpg", "grayscale": False},
        {"name": "5a. Plane01 in Sky (Color)", "path": "test_plane.jpg", "grayscale": False},
        {"name": "5b. Plane01 in Sky (Grayscale)", "path": "test_plane.jpg", "grayscale": True}, # Re-uses the same image
        {"name": "6a. Plane02 in Sky (Color)", "path": "test_plane02.jpg", "grayscale": False},
        {"name": "6b. Plane02 in Sky (Grayscale)", "path": "test_plane02.jpg", "grayscale": True} # Re-uses the same image
    ]
    
    print("\n" + "="*50)
    print("ADVERSARIAL CONTEXT BIAS REPORT")
    print("="*50)
    
    for test in tests:
        pred_class, confidence = analyze_image(test["path"], model, CLASS_NAMES, test["grayscale"])
        
        if pred_class is None:
            print(f"{test['name']:<30} | [FILE NOT FOUND: {test['path']}]")
        else:
            print(f"{test['name']:<30} | Predicted: {pred_class:<15} | Confidence: {confidence:.2f}%")
            
    print("="*50)