import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Define sports classes (you may need to replace these with your own dataset)
classes = [ "air hockey", "amputee football", "archery", "arm wrestling", "axe throwing",
    "balance beam", "barrel racing", "baseball", "basketball", "baton twirling",
    "bike polo", "billiards", "bmx", "bobsled", "bowling", "boxing", "bull riding",
    "bungee jumping", "canoe slalom", "cheerleading", "chuckwagon racing", "cricket",
    "croquet", "curling", "disc golf", "fencing", "field hockey", "figure skating men",
    "figure skating pairs", "figure skating women", "fly fishing", "football",
    "formula 1 racing", "frisbee", "gaga", "giant slalom", "golf", "hammer throw",
    "hang gliding", "harness racing", "high jump", "hockey", "horse jumping",
    "horse racing", "horseshoe pitching", "hurdles", "hydroplane racing", "ice climbing",
    "ice yachting", "jai alai", "javelin", "jousting", "judo", "lacrosse", "log rolling",
    "luge", "motorcycle racing", "mushing", "nascar racing", "olympic wrestling",
    "parallel bars", "pole climbing", "pole dancing", "pole vault", "polo", "pommel horse",
    "rings", "rock climbing", "roller derby", "rollerblade racing", "rowing", "rugby",
    "sailboat racing", "shot put", "shuffleboard", "sidecar racing", "ski jumping",
    "sky surfing", "skydiving", "snowboarding", "snowmobile racing", "speed skating",
    "steer wrestling", "sumo wrestling", "surfing", "swimming", "table tennis", "tennis",
    "track bicycle", "trapeze", "tug of war", "ultimate", "uneven bars", "volleyball",
    "water cycling", "water polo", "weightlifting", "wheelchair basketball",
    "wheelchair racing", "wingsuit flying"]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_sport(image):
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return classes[predicted_class]

def main():
    st.title("Sports Name Prediction with Deep Learning")

    uploaded_file = st.file_uploader("Choose a sports logo image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = preprocess_image(uploaded_file)

        # Classify using the pre-trained model
        result = classify_sport(image)

        # Display result
        st.write(f"**Predicted Sport:** {result}")

if __name__ == "__main__":
    main()
