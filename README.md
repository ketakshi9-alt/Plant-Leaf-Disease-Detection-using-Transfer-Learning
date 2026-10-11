# 🌿 Plant Leaf Disease Detection using Transfer Learning

A deep learning web application that detects whether a tomato plant leaf is **healthy** or affected by diseases such as **Bacterial Spot**, **Early Blight**, **Late Blight**, **Septoria Leaf Spot**, and **Yellow Leaf Curl Virus**.

The project uses **Transfer Learning** with a pre-trained **MobileNetV2** model and provides a simple **Streamlit** web interface where users can upload a leaf image and get an instant prediction with a confidence score.

---

## 📌 Project Overview

Farmers often find it difficult to identify plant diseases at an early stage. This project allows users to upload a tomato leaf image and automatically classify it into one of the following categories:

- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Septoria Leaf Spot
- Yellow Leaf Curl Virus

The system uses a pre-trained MobileNetV2 model as a feature extractor and trains a custom classification layer on plant leaf images.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Streamlit
- NumPy
- Pillow
- Matplotlib
- sklearn

---

## 🚀 Features

- Upload tomato leaf image
- Detect healthy or diseased leaf
- Predict disease class with confidence score
- Show class-wise prediction probabilities
- Training and validation accuracy/loss graphs
- Confusion matrix and classification report
- Simple and interactive Streamlit UI
- Uses transfer learning for better accuracy with less training time

---

## 🗂️ Project Structure

```text
project/
│
├── .venv/                  # Virtual environment
├── .archive/               # Archived/backup files
├── app.py                  # Streamlit web application
├── class_names.json        # Disease class labels
├── Model.ipynb             # Dataset loading, model training and evaluation
└── plant_model.keras       # Trained deep learning model
```

---

## 📁 Dataset

The dataset contains tomato leaf images divided into six disease classes.

```text
Tomato/
├── Train/
│   ├── Bacterial Spot/
│   ├── Early Blight/
│   ├── Healthy/
│   ├── Late Blight/
│   ├── Septoria Leaf Spot/
│   └── Yellow Leaf Curl Virus/
│
├── Val/
│   ├── Bacterial Spot/
│   ├── Early Blight/
│   ├── Healthy/
│   ├── Late Blight/
│   ├── Septoria Leaf Spot/
│   └── Yellow Leaf Curl Virus/
│
└── Test/
    ├── Bacterial Spot/
    ├── Early Blight/
    ├── Healthy/
    ├── Late Blight/
    ├── Septoria Leaf Spot/
    └── Yellow Leaf Curl Virus/
```

The model classifies a tomato leaf image into one of the following categories:

- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Septoria Leaf Spot
- Yellow Leaf Curl Virus

## 🧪 Model Architecture

The model uses **Transfer Learning**:

1. Load MobileNetV2 pre-trained on ImageNet
2. Remove its original classification layer
3. Freeze the pre-trained layers
4. Add custom classification layers
5. Train only the new layers on plant leaf images

```text
Input Image (224 x 224 x 3)
        ↓
Data Augmentation
        ↓
MobileNetV2 Pretrained Base
        ↓
Global Average Pooling
        ↓
Dropout Layer
        ↓
Dense Output Layer (Softmax)
        ↓
Predicted Disease Class
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone 
cd project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 requirements.txt

```text
tensorflow
streamlit
numpy
matplotlib
scikit-learn
```

---

## 🏋️ Model Training

Open and run the Jupyter Notebook:

```bash
jupyter notebook Model.ipynb
```

The notebook performs the following steps:

1. Loads the plant leaf dataset
2. Applies data augmentation
3. Loads MobileNetV2 pretrained model
4. Builds a custom classification model
5. Trains the model
6. Evaluates accuracy, precision, recall and F1-score
7. Plots training vs validation graphs
8. Saves the trained model as `plant_model.keras`
9. Saves class labels in `class_names.json`

---

## 🖥️ Run Streamlit App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Upload a tomato leaf image and click **Predict Disease**.

---

## 🖼️ How to Use

1. Run the Streamlit application.
2. Upload a tomato leaf image.
3. Click on **Predict Disease**.
4. View the predicted disease and confidence score.
5. Check class-wise prediction probabilities.

---

## 📈 Sample Output

```text
Prediction: Tomato - Early Blight
Confidence: 89.45%
```

## 🌱 Diseases Detected

### Healthy Leaf
A normal green tomato leaf without disease symptoms.

### Early Blight
A fungal disease that causes dark brown spots with concentric rings on leaves.

### Late Blight
A serious disease that causes dark water-soaked lesions and can spread quickly in humid conditions.

---