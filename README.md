#  ML Model for Freshness Detection of Fruits and Vegetables

## 📌 Overview
This project presents a Machine Learning-based system for detecting the freshness of fruits and vegetables using image processing techniques. The system automates quality inspection by analyzing visual features such as color, texture, and shape, reducing human effort and improving accuracy.

## 🎯 Problem Statement
Traditional freshness detection relies on manual inspection, which is subjective, inconsistent, and error-prone. This project addresses the need for an automated, scalable, and reliable system to classify produce as **fresh or stale** using image data.

## 🚀 Key Features
- Automated freshness detection using image input
- Classification of fruits and vegetables into fresh or stale categories
- Real-time prediction system with GUI interface
- Approximately **85% classification accuracy**
- Integrated Exploratory Data Analysis (EDA) tools
- User-friendly interface for non-technical users

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Deep Learning:** MobileNetV2 (Feature Extraction)  
- **Machine Learning:** Random Forest Classifier  
- **Libraries:** OpenCV, NumPy, Scikit-learn, TensorFlow/Keras  
- **GUI:** Tkinter  
- **Visualization:** Matplotlib, Seaborn  

## ⚙️ System Architecture
1. **Image Input**
2. **Preprocessing (Resize, Normalize, Orientation Fix)**
3. **Feature Extraction using MobileNetV2**
4. **Classification using Random Forest**
5. **Prediction Output (Fresh / Stale)**

## 🔄 Project Workflow (Step-by-Step)

### 1. Data Collection
- Collected labeled dataset of fruits and vegetables (fresh & stale)

### 2. Data Preprocessing
- Image resizing (224x224)
- EXIF orientation correction
- Normalization of pixel values

### 3. Feature Extraction
- Used **MobileNetV2** to extract deep image features

### 4. Model Training
- Trained **Random Forest Classifier** on extracted features

### 5. Model Evaluation
- Evaluated using accuracy metrics
- Achieved ~**85% accuracy**

### 6. Prediction System
- Input image → Model → Output (Fresh / Stale)

### 7. GUI Integration
- Built interactive interface using Tkinter
- Allows:
  - Data loading
  - Training
  - Visualization
  - Prediction

## 📊 Exploratory Data Analysis (EDA)
- Brightness Histogram Analysis
- PCA Visualization (feature distribution)
- Class distribution charts

## 📂 Project Structure

freshness-detection-ml/
│
├── 
│   ├── train/
│   ├── test/
│
└── main.py   

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/rajeshbathini53/Freshness-Detection-of-Fruits-and-Vegetables-.git
```

2. Navigate to project folder

  cd Freshness Detection of fruits and vegetables

3. Install dependencies

  pip install -r requirements.txt

4. Run the application
  
  python main.py
  
5. Install dependencies

  pip install -r requirements.txt

6. Run the application

  python main.py

📈 Results
1.Achieved approximately 85% accuracy
2.Successfully detects freshness using image-based classification
3.Provides real-time predictions via GUI
