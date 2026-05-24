import os
import time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import EfficientNet_B3_Weights
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. Model Setup ---
def build_interpretable_model(num_classes: int):
    weights = EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)
    
    for param in model.features.parameters():
        param.requires_grad = False
            
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

# --- 2. Data Pipeline ---
def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img_full:
            if img_full.mode not in ['RGB'] or img_full.format in ['WEBP', 'CMYK']:
                return False
        return True
    except Exception:
        return False

def get_dataloaders(data_dir: str, batch_size: int = 16):
    IMG_SIZE = 300 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(320), 
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Training'), transform=train_transforms, is_valid_file=is_valid_image)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Validation'), transform=val_transforms, is_valid_file=is_valid_image)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, train_dataset.classes

# --- 3. Plotting Function ---
def save_training_report(history):
    """Generates and saves a matplotlib figure of the training curves."""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_report.png', dpi=300)
    print("\n--> Training report saved as 'training_report.png'")

# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, num_epochs=50, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler('cuda') 
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Dictionary to track metrics for the report
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_interpretable_model.pth')
            print("--> Checkpoint saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
                
    # Generate the visual report at the end
    save_training_report(history)

if __name__ == "__main__":
    # UPDATE THIS PATH
    DATA_DIR = r"C:\Users\sameh\Desktop\XAI\vehicle" 
    
    print("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR)
    
    print(f"Building model for {len(class_names)} classes...")
    model = build_interpretable_model(num_classes=len(class_names))
    
    print("Starting training...")
    train_model(model, train_loader, val_loader)