# ImageClassification_With_DeepLearning
# why do we move to deep Learning from Machine Learning?
Deep learning, a type of machine learning, uses deep neural networks with multiple layers. It's favored because these networks can automatically learn intricate patterns from data, making them especially good at tasks like image and speech recognition. The complexity of deep learning enables better handling of large and unstructured datasets, outperforming traditional machine learning in certain applications.

These are just training codes that are done by me on Google Colab. I have worked so many models of transfer learning. Go to these all files , run on google colab , that may help you a lot , we are trying these are models , coz we want to see the best model and higher accurator. That gives much accuracy on training as well as on validation, That's why are trying these all models.

At the end of running the code on colab , save the model with .h5 .That must keep in mind Like; model.save("anyName.h5")

# streamlit
I have uploaded the steamlit codes for concern datasets from kaggle on the following transfer learning models.
There are few folders and in those folders there are streamlit codes and the related pictures of the concern predicted dataset.
# what you have to do?
You have the run the codes first, that are present in the AI and All_Models_Keras folders.
after that , at the end of the code you should save the model with .h5 extension.
Like, model.save("anyName.h5")

# Furthermore , download the save model where the streamlit codes are present.
Then open the python code whiich is saved with .py extension.

After that do the following as I did in the below in coding...
# Code
import numpy as np
import cv2 
import streamlit as st
from keras.utils import load_img, img_to_array
from keras.models import load_model

model=load_model('Breast_ultrosound.h5')
st.title("BREAST_ULTRASOUND")
class_labels=[
    'benign',
    'malignant',
    'normal'
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
      st.write(f' {d_predicted}')

Instead of Breast_ultrasound.h5 , save your download file , then run the code for testing the images ,whether it works perfect or not.

then do some practices by yourselves , to earn the higher accuracy.
I hope it helps you a lot.
