import numpy as np
import cv2 
import streamlit as st
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

model = load_model('real-life-violence-situations.h5')
st.title("Violation App")
class_labels = ['non_violence', 'violence']

uploader = st.file_uploader('Select image', type=('jpg', 'png'))

if uploader is not None:
    image = load_img(uploader, target_size=(224, 224), color_mode='rgb')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_labels[class_index]

    st.subheader("Prediction:")
    st.write(f' {predicted_class}')

    if predicted_class == 'violence':
        st.warning("Violence Detected!")
    else:
        st.success("Non-violence Detected!")
