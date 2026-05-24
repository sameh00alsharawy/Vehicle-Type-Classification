import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from captum.attr import visualization as viz

# 1. Rebuild the exact same model architecture
def load_trained_model(model_path, num_classes):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    # Load the weights we just trained
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # Set to evaluation mode
    return model

def generate_hero_image(image_path, model, class_names):
    # Prepare the image exactly as the model expects
    transform = transforms.Compose([
        transforms.Resize(320), 
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0) # Add batch dimension
    
    # 1. Make a prediction
    outputs = model(input_tensor)
    predicted_idx = torch.argmax(outputs).item()
    predicted_class = class_names[predicted_idx]
    
    # 2. XAI Setup: Hook GradCAM into the final feature layer of EfficientNet
    target_layer = model.features[-1]
    grad_cam = LayerGradCam(model, target_layer)
    
    # 3. Generate the attribution heatmap for the predicted class
    attributions = grad_cam.attribute(input_tensor, target=predicted_idx)
    
    # Resize the heatmap to match the original image size
    attributions_resized = LayerGradCam.interpolate(attributions, (300, 300))
    
    # Format data for Captum's visualization library
    img_viz = input_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    # Un-normalize for display
    img_viz = img_viz * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_viz = np.clip(img_viz, 0, 1)
    
    attr_viz = attributions_resized.squeeze().detach().numpy()
    
    # 4. Create the side-by-side visual
    fig, axis = viz.visualize_image_attr_multiple(
        attr_viz,
        img_viz,
        methods=["original_image", "blended_heat_map"],
        signs=["all", "positive"],
        titles=["Original Image", f"Grad-CAM (Prediction: {predicted_class})"],
        show_colorbar=True,
        fig_size=(10, 5)
    )
    plt.show()

if __name__ == "__main__":
    # UPDATE THESE
    MODEL_WEIGHTS = "best_interpretable_model.pth"
    TEST_IMAGE_PATH = r"C:\Users\sameh\Desktop\XAI\vehicle\Testing\Trains\Train (720).jpg" # Pick a single car image
    CLASS_NAMES = ['Auto', 'Bike', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train'] # Ensure order matches train_loader
    
    print("Loading model...")
    model = load_trained_model(MODEL_WEIGHTS, len(CLASS_NAMES))
    
    print("Generating Explainability visual...")
    generate_hero_image(TEST_IMAGE_PATH, model, CLASS_NAMES)