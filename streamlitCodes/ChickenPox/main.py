import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model architecture and weights
model = tf.keras.models.load_model('Chicken.h5')

# Load class names (replace with your actual class names)
sports_categories = ["Chicken Pox", "Monkey Pox", "Mesay", "Normal"]

# Title and description
st.title("Deep Learning Model Prediction App")
st.write("Upload an image to get a prediction and its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")


def preprocess_image(image):
    # Resize to a fixed size
    image = tf.image.resize(image, (224, 224))

    # Normalize pixel values
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)

    return image


if uploaded_file is not None:
    # Process image (if applicable)
    image = Image.open(uploaded_file).convert('RGB')
    image = np.array(image) / 255.0  # Normalize pixels

    # Preprocess image according to your model's requirements
    preprocessed_image = preprocess_image(image)  # Replace with your preprocessing function

    # Make prediction
    prediction = model.predict(preprocessed_image[np.newaxis, ...])

    # Display results
    st.image(image, caption="Uploaded Image", width=300)
    st.subheader("Predictions:")

