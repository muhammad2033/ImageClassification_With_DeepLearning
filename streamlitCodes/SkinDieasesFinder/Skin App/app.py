import numpy as np
import cv2 
import streamlit as st
from keras.utils import load_img, img_to_array
from keras.models import load_model

model=load_model('SModel.h5')
st.title("Skin Disease Diagnostic App")
class_labels=[
    'Skin Diseases',
    'Acne',
    'Actinic Carcinoma',
    'Atopic Dermatitis',
    'Bullous Disease',
    'Cellulitis',
    'Eczema',
    'Drug Eruptions',
    'Herpes HPV',
    'Light Diseases',
    'Lupus',
    'Melanoma',
    'Poison Ivy',
    'Psoriasis',
    'Benign Tumors',
    'Systemic Disease',
    'Ringworm',
    'Urticarial Hives',
    'Vascular Tumors',
    'Vasculitis',
    'Viral Infections'
]
uploader = st.file_uploader('Select image', type=('jpg', 'png'))
if uploader is not None:
    image = load_img(uploader,target_size=(224,224),color_mode='rgb')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.subheader("Prediction:")
    d_predicted=class_labels[class_index]
    st.write(f'Your Skin Disease is {d_predicted}')

    

    
    
    
    
    
    












    
  