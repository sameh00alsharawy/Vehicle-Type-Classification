import os
import csv
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

# --- 2. Test Data Pipeline ---
def get_test_dataloader(data_dir: str, batch_size: int = 16):
    IMG_SIZE = 300 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([
        transforms.Resize(320), 
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Testing'), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return test_loader, test_dataset

# --- 3. Evaluation Logic ---
def evaluate_model(model, test_loader, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_names = test_dataset.classes
    
    all_preds = []
    all_labels = []
    misclassified_data = [] # List to hold our error report
    
    global_idx = 0 # Tracker to match predictions to filenames
    
    print(f"Evaluating on device: {device}")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Running Test Set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Convert to CPU for scikit-learn
            preds_cpu = predicted.cpu().numpy()
            labels_cpu = labels.numpy()
            
            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)
            
            # --- MISCLASSIFICATION TRACKER ---
            for i in range(len(labels_cpu)):
                if preds_cpu[i] != labels_cpu[i]:
                    # Grab the original file path from the dataset using our global index
                    file_path = test_dataset.samples[global_idx][0]
                    file_name = os.path.basename(file_path)
                    
                    true_class = class_names[labels_cpu[i]]
                    pred_class = class_names[preds_cpu[i]]
                    
                    misclassified_data.append([file_name, true_class, pred_class])
                
                global_idx += 1
            
    # 1. Generate Text Report 
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # 2. Generate Confusion Matrix Visual
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='d', ax=plt.gca())
    plt.title('Test Dataset Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n--> Saved visual confusion matrix to 'confusion_matrix.png'")
    
    # 3. Save Misclassified Report to CSV
    csv_file = 'misclassified_report.csv'
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'True Class', 'Predicted Class'])
        writer.writerows(misclassified_data)
    print(f"--> Saved misclassification report to '{csv_file}' ({len(misclassified_data)} errors logged)")

if __name__ == "__main__":
    # UPDATE THESE PATHS
    DATA_DIR = r"C:\Users\sameh\Desktop\XAI\vehicle" 
    MODEL_WEIGHTS = "best_interpretable_model.pth"
    
    print("Loading test data...")
    test_loader, test_dataset = get_test_dataloader(DATA_DIR)
    
    print("Loading best model weights...")
    model = load_trained_model(MODEL_WEIGHTS, len(test_dataset.classes))
    
    print("Starting evaluation...")
    evaluate_model(model, test_loader, test_dataset)