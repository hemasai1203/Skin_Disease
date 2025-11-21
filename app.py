import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tempfile import NamedTemporaryFile

# Load Model
model = load_model('./Model/BC.h5', compile=False)

# Label Dictionary
lab = {0: 'Acne', 1: 'Carcinoma', 2: 'Eczema', 3: 'Keratosis', 4: 'Milia', 5: 'Rosacea'}

# Image Processing Function
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Correct target size
    img = img_to_array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)  # Get class with highest probability
    y = int(y_class[0])  # Extract class index
    res = lab[y]  # Map to label
    return res

# Streamlit Application
def run():
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((350, 350))
    st.image(img1, use_column_width=False)
    st.title("Skin conditions classification based on vitamin deficiency using CNN 💉🩺")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based on "six skin conditions datasets"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Skin", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file, use_column_width=False)
        
        # Save uploaded file temporarily
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(img_file.getbuffer())
        save_image_path = temp_file.name

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted skin condition is: " + result)
        else:
            st.success("you given wrong image")


# Run the app
run()
