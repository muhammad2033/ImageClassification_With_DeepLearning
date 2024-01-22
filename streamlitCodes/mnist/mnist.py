import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# Load the pre-trained MNIST model
model_path = 'mnist.h5'  # Replace with the path to your pre-trained model
mnist_model = load_model(model_path)

def preprocess_image(image):
    # Preprocess the image to match the model input format
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit UI
st.title("MNIST Digit Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and classify the digit
    preprocessed_image = preprocess_image(image)
    prediction = mnist_model.predict(preprocessed_image)

    # Display the classification result
    predicted_label = np.argmax(prediction)
    st.write(f"Prediction: {predicted_label}")
