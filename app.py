import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('brain_tumor.h5')

# Class labels — update if you have different labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"### 🧾 Predicted Tumor Type: {predicted_class}")