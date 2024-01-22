import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image

# Load the Fashion MNIST model
model_path = 'model.h5'
model = load_model(model_path)

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Streamlit app
st.title("Fashion MNIST Classifier")
st.sidebar.title("Upload Image")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image in the main section
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match the model's expected sizing
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for prediction

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display the predicted class label in a styled box
    st.success(f"Prediction: {class_labels[predicted_class]}")

    # Display the probability scores for each class
    st.subheader("Class Probabilities:")
    probs_df = pd.DataFrame({'Class': class_labels, 'Probability': prediction[0]})
    st.bar_chart(probs_df.set_index('Class'))
