![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classic-green)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange)
![Status](https://img.shields.io/badge/Status-Academic%20Project-brightgreen)

# 🌿 AgriVision Pro – Plant Disease Detection System

AgriVision Pro is a desktop-based machine learning application designed to automatically detect and classify plant diseases from leaf images. The system uses a hybrid approach combining unsupervised image segmentation and supervised classification.

## 🖥️ Application Screenshots

### 🔹 Main Dashboard
![Main Dashboard](Screenshots/dashboard.png)

### 🔹 Disease Prediction Result
![Prediction Output](Screenshots/prediction.png)



## 🔍 Methodology
1. **Image Preprocessing**
   - Resize images to 256×256 pixels

2. **Segmentation**
   - K-Means clustering (k=2) to isolate diseased regions

3. **Feature Extraction (10 Features)**
   - Color: Mean & Std of RGB (6)
   - Texture: GLCM Contrast, Correlation, Energy (3)
   - Shape: Diseased area ratio (1)

4. **Classification**
   - Random Forest Classifier (100 trees)

5. **GUI**
   - Desktop application built using CustomTkinter

## 🧠 Model
- Algorithm: Random Forest Classifier
- Training/Test Split: 80/20
- Dataset: PlantVillage
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt



