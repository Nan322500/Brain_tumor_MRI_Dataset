
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('brain_tumor_model.h5')

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor Classification")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: **{predicted_class}**")
