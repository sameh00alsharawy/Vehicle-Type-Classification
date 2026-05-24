***

# Indian Vehicle Classification - IVDAP 2025

##  Project Overview
This project focuses on classifying vehicles into 7 distinct categories using deep learning. Developed for the **IVPAD 2025** course, the model is designed to support smart city applications, such as automated roadside traffic monitoring and traffic flow optimization.

By accurately identifying different types of vehicles (including cars, motorcycles, and public transport), this system can be integrated into real-time monitoring pipelines to enhance urban mobility and safety.

##  Team Members
* **Ibrahim Abdellatif** (D1800069)
* **Sameh Alshaarawy** (D18000042)
* **Zeyad Gharaf** (D18000059)
* **Supervisor:** Prof. Luisa Verdoliva

##  Dataset
The project utilizes a custom dataset consisting of **5,600 images** evenly distributed across 7 classes to ensure a balanced training process. 
* **Total Images:** 5,600
* **Images per Class:** 800
* **Classes:** 1. Auto
    2. Bike
    3. Car
    4. Motorcycle
    5. Plane
    6. Ship
    7. Train

##  Methodology & Architecture

### 1. Data Preprocessing & Cleaning
Real-world image datasets often contain corrupted or unsupported files. The pipeline includes a robust cleaning script utilizing the `PIL` library to iterate through pandas DataFrames and remove:
* Corrupted `.jpg` and `.png` files.
* Unsupported `.webp` formats.
* Images with CMYK color profiles that disrupt standard RGB processing.

### 2. Data Augmentation
To prevent overfitting and improve the model's ability to generalize to unseen data, a TensorFlow `Sequential` data augmentation pipeline was implemented, applying:
* Random Rotation
* Random Zoom
* Random Horizontal Flips

### 3. Model Architecture
The core of the classification system relies on **EfficientNetB3**:
* **Transfer Learning:** The model utilizes pre-trained `ImageNet` weights.
* **Fine-Tuning:** The base model's top layer is excluded (`include_top=False`), and a custom dense classification head is trained specifically on our 7 vehicle categories.

##  Prerequisites & Setup

To run the Jupyter Notebook, you will need the following dependencies installed in your Python environment (or Google Colab):

```bash
pip install tensorflow
pip install keras-cv
pip install pandas
pip install pillow
pip install easy-cv-dataset
```

##  How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/indian-vehicle-classification.git
    cd indian-vehicle-classification
    ```
2.  **Dataset Preparation:**
    * Ensure the dataset (`vehicle.zip`) is available. 
    * If using Google Colab, upload the zip file to your Google Drive and ensure the Drive mount paths in the notebook (`/content/drive/...`) match your directory structure.
3.  **Run the Notebook:**
    * Open `Vehicle_Classification_Model_IVDAP.ipynb` in Google Colab or your local Jupyter environment.
    * Execute the cells sequentially to mount the drive, extract the data, clean the dataset, train the EfficientNetB3 model, and evaluate its performance.

##  Repository Structure
* `Vehicle_Classification_Model_IVDAP.ipynb`: Main Python notebook containing the data pipeline, model training, and evaluation code.
* `IVDAP Project Presentation.pdf`: Slide deck detailing the project's aims, dataset, methodology, and real-case scenarios.

##  Acknowledgments
Special thanks to **Prof. Luisa Verdoliva** for supervising this project as part of the IVPAD 2025 curriculum.

*** *Tip: Don't forget to replace `your-username` in the "How to Run" section with your actual GitHub username! Let me know if you'd like to add any specific model accuracy metrics or evaluation charts to the README.*
