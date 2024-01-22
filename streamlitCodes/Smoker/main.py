import numpy as np
import cv2 
import streamlit as st
from keras.utils import load_img, img_to_array
from keras.models import load_model

# Define the custom layer (if not already defined)
# Example: Assuming LayerScale is a custom layer
class LayerScale:
    # Your layer implementation goes here
    pass

# Load the model with the custom_objects parameter
model = load_model('smoke (1).h5', custom_objects={'LayerScale': LayerScale})

st.title("Smoker Recognition App")
class_labels = ['smoker', 'non-smoker']

uploader = st.file_uploader('Select image', type=('jpg', 'png'))
if uploader is not None:
    image = load_img(uploader, target_size=(224, 224), color_mode='rgb')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.subheader("Prediction:")
    d_predicted = class_labels[class_index]
    st.write(f' {d_predicted}')
