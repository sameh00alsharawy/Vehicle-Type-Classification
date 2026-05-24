
# Indian Vehicle Classification & AI Safety Audit (IVDAP 2025)

## 📌 Project Overview

This project focuses on classifying vehicles into 7 distinct categories using deep learning. Initially developed for the **IVDAP 2025** course, the codebase has been entirely refactored in **PyTorch** to serve as a framework for **Explainable AI (XAI) and Model Auditing**. *(Details regarding the original Keras implementation can be found in `readme_old.md`).*

By accurately identifying different types of vehicles, this system can be integrated into real-time monitoring pipelines. More importantly, this repository demonstrates how to use post-hoc interpretability tools (Grad-CAM) and adversarial testing to mathematically audit neural networks for contextual bias and spurious correlations.

## 📊 Dataset

The project utilizes a custom dataset consisting of **5,600 images** evenly distributed across 7 classes:
`Auto Rickshaws` | `Bikes` | `Cars` | `Motorcycles` | `Planes` | `Ships` | `Trains`

## 🧠 Methodology & Architecture

### 1. Data Preprocessing & Pipeline

The pipeline utilizes a robust, lazy-loading PyTorch `DataLoader` optimized for GPU data transfers (`pin_memory=True`). It includes on-the-fly cleaning using `ImageFolder`'s `is_valid_file` parameter to dynamically reject:

* Corrupted `.jpg` and `.png` headers.
* Unsupported `.webp` formats.
* Images with CMYK color profiles that disrupt standard RGB tensors.

### 2. Data Augmentation

To prevent overfitting and improve generalization, `torchvision.transforms` were applied to the training set:

* Random Rotation (±20 degrees)
* Random Resized Crops (Scale: 0.7 - 1.3)
* Random Horizontal Flips

### 3. Model Architecture (EfficientNet-B3)

* **Transfer Learning:** The model utilizes modern `EfficientNet_B3_Weights.DEFAULT` (ImageNet).
* **Linear Probing:** The entire convolutional feature extractor (`model.features`) was frozen to prevent catastrophic forgetting.
* **Custom Head:** A custom dense classification head with Dropout (`p=0.3`) was attached and trained specifically on our 7 vehicle categories.

## ⚙️ Prerequisites & Setup

This project requires Python 3.11+ and an NVIDIA GPU (CUDA 11.8+ recommended).

```bash
# 1. Create and activate a Conda environment
conda create -n safety_xai python=3.11 -y
conda activate safety_xai

# 2. Install PyTorch (CUDA 12.4) and scikit-learn
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 scikit-learn -c pytorch -c nvidia -y

# 3. Install XAI and plotting utilities
pip install captum matplotlib opencv-python tqdm

```

## 🚀 Training & Performance

Training was executed on an **NVIDIA RTX 4060 Laptop GPU** utilizing Automatic Mixed Precision (AMP) to optimize VRAM usage.

* **Epochs:** 50
* **Batch Size:** 16
* **Loss Function:** Cross-Entropy Loss (optimizing for model confidence/calibration)
* **Optimizer:** AdamW (Learning Rate: `1e-4`, Weight Decay: `1e-4`)
* **Schedulers:** `ReduceLROnPlateau` and Early Stopping tracking Validation Loss.

### Results

* **Test Dataset Accuracy:** `0.9881`

*(Insert `training_report.png` here)*
*(Insert `confusion_matrix.png` here)*

---

## 🔍 Model Auditing & Failure Mode Analysis (XAI)

Despite achieving nearly 99% accuracy, high accuracy on identically distributed data can mask critical flaws. To ensure the model is safe for real-world deployment, a **Post-Hoc Model Audit** was performed on the misclassified test images using `LayerGradCam` hooked into the final convolutional layer of EfficientNet.

The goal was to formulate hypotheses regarding the failure modes, design ablation experiments to test those hypotheses, and provide engineering recommendations—all without retraining the model to strictly prevent data leakage.

### A. Context Bias (The "Clever Hans" Effect)

The model frequently hallucinated vehicles based on background elements.

* **Image 698 (Motorcycle $\rightarrow$ Train):** Grad-CAM highlighted background fencing. The model falsely learned that repetitive vertical lines equate to train tracks.
* **Image 769 (Ship $\rightarrow$ Car):** Grad-CAM focused entirely on the asphalt beneath a dry-docked boat, assuming `Asphalt = Car`.
* **Image 800 (Ship $\rightarrow$ Plane):** Grad-CAM highlighted vast blue water, misinterpreting it as blue sky.

### B. Multi-Object & Framing Violations

* **Image 793 (Auto $\rightarrow$ Motorcycle):** The model correctly identified a motorcycle in the background. The classification task itself is flawed when multiple objects exist in a single frame.
* **Image 773 (Ship $\rightarrow$ Plane):** The aircraft carrier literally contained airplanes, which the model correctly focused on.

### C. Dataset Noise (Labeling Errors)

* **Image 762 (Motorcycle $\rightarrow$ Bike):** The image was actually a bicycle, but the ground truth was incorrectly labeled "Motorcycle." Grad-CAM perfectly highlighted the bicycle frame, proving the AI extracted the correct features despite the human annotation error.

*(Insert a composite image of 3-4 side-by-side Grad-CAM error heatmaps here)*

---

## 🧪 Adversarial Hypothesis Testing

To quantify the extent of the Context Bias, an adversarial script (`test_bias.py`) was developed to feed the model pure backgrounds (no vehicles) and perform color ablations.

```text
==================================================
ADVERSARIAL CONTEXT BIAS REPORT
==================================================
1. Just Sky01                  | Predicted: Planes          | Confidence: 64.35%
2. Just Sky02                  | Predicted: Ships           | Confidence: 50.32%
2. Just Sea                    | Predicted: Ships           | Confidence: 94.47%
3. Train Tracks (No Train)     | Predicted: Trains          | Confidence: 96.85%
4. Vertical Fences (Not Tracks)| Predicted: Trains          | Confidence: 71.35%
5a. Plane01 in Sky (Color)     | Predicted: Planes          | Confidence: 99.98%
5b. Plane01 (Grayscale)        | Predicted: Planes          | Confidence: 99.84%
6a. Plane02 in Sky (Color)     | Predicted: Planes          | Confidence: 78.95%
6b. Plane02 (Grayscale)        | Predicted: Planes          | Confidence: 87.83%
==================================================

```

### Conclusions from Adversarial Testing:

1. **Confirmation of Scene Bias:** The model is almost completely certain that water equals a Ship (94.4%) and rails equal a Train (96.8%). It is so biased toward geometric lines that it hallucinates trains when looking at a wooden fence (71.3%).
2. **Disproving the Color Hypothesis:** It was initially hypothesized that the model failed on Image 708 (a grayscale plane) due to a lack of blue sky. The ablation data (`99.98% -> 99.84%`) empirically disproves this. Furthermore, in Test 6, grayscale *improved* confidence (`78.95% -> 87.83%`).
3. **Refined Hypothesis:** The model successfully relies on the geometric silhouette (edges/shapes) of airplanes regardless of color space. Image 708 likely failed because it was photographed from a ground-level angle with prominent circular landing gear, causing an overlapping feature conflict with the "Motorcycle" class.

## 🛠️ Recommendations for Future Iterations

Based on this audit, the following architectural changes are recommended prior to real-world deployment:

1. **Transition to Object Detection (YOLO/Faster R-CNN):** To resolve multi-object framing violations (Image 793).
2. **Targeted Data Augmentation:** Introduce aggressive background-altering augmentations (e.g., boats on trailers, planes in hangars) to break the `Ship = Water` spurious correlation.
3. **Automated Label Auditing:** Implement confidence-scoring scripts to flag low-confidence training data for human re-annotation to resolve label noise (Image 762).

---

## 💻 How to Run

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/indian-vehicle-classification.git
cd indian-vehicle-classification

```


2. **Dataset Preparation:**
* Download the dataset and extract it directly into the project root folder.
* Ensure the folder structure is: `vehicle/Training`, `vehicle/Validation`, `vehicle/Testing`.


3. **Execution Order:**
* Run `python train.py` to compile the model and train the custom classification head.
* Run `python evaluate.py` to generate the test metrics and confusion matrix.
* Run `python interpret.py` to generate the XAI heatmaps.
* Run `python test_bias.py` to run the adversarial context audit.



## 📁 Repository Structure

* `train.py`: Core training loop, data augmentation, and AMP scaling.
* `evaluate.py`: Statistical testing and CSV error reporting.
* `interpret.py`: Captum LayerGradCam implementation.
* `analyze_errors.py`: Batch script to generate heatmaps for all misclassified images.
* `test_bias.py`: Adversarial inference script for hypothesis testing.
* `Vehicle_Classification_Model_IVDAP.ipynb`: Original Keras implementation (Legacy).
* `IVDAP Project Presentation.pdf`: Slide deck outlining the original project goals.

## 🎓 Acknowledgments

Special thanks to **Prof. Luisa Verdoliva** for supervising the original foundation of this project as part of the IVDAP 2025 curriculum.
