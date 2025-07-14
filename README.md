# Brain Tumor MRI Classification

This project is a **Brain Tumor Classification** application built using a Convolutional Neural Network (CNN) to classify MRI images into tumor types: Glioma, Meningioma, No Tumor, and Pituitary tumor. The app is implemented using **TensorFlow** and deployed with **Streamlit** for easy interaction.

---

## Project Overview

Brain tumor detection is crucial for early diagnosis and treatment. This model takes an MRI scan as input and predicts the type of brain tumor present. It aims to assist medical professionals in fast preliminary screening.

---

## Dataset

The dataset used for training is a public Brain MRI images dataset with labeled tumor types. (Include link here if available)

---

## Model

- CNN-based deep learning model
- Trained on MRI images resized to 224x224
- Achieved good accuracy on validation data

---

## Files in This Repository

- `app.py`: Streamlit web app for uploading MRI images and getting predictions  
- `brain_tumor_model.h5`: Trained TensorFlow model file  
- `requirements.txt`: Python dependencies  
- `runtime.txt`: Python version for deployment  
- `README.md`: This file  

---

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nan322500/Brain_tumor_MRI_Dataset.git
   cd Brain_tumor_MRI_Dataset
