import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_lottie import st_lottie
import requests
import cv2
from PIL import Image, ImageOps

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource(show_spinner="loading...")
def load_lottie():
    r = requests.get("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")
    if r.status_code != 200:
        return None
    return r.json()
with st.sidebar:
    lottie_json = load_lottie()
    st_lottie = st_lottie(lottie_json,speed=1,loop=True,quality="high",height=400, width=300)

@st.cache_resource(show_spinner="Loading..", experimental_allow_widgets=True)
def model_():
    interpreter = tf.lite.Interpreter(model_path="pages/../converted_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = model_()

# Get input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

col1, col2, col3 = st.columns([3 ,11, 3])

class_indices = {0: 'Mild Dementia', 1: 'Moderate Dementia',
                 2: 'Non Demented', 3: 'Very mild Dementia'}

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    img = np.asarray(img).astype(np.float32)
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    print("Top prediction:", output_data.argmax())

    return output_data.argmax()

with col2:
    st.title("Discover Your Outcome")
    file_ = st.file_uploader("Please upload the Brain X-ray image", type=['jpg','png','jpeg'])
    st.markdown("""___""")


if file_ is None:
    st.text(" ")
else:
    with col2:
        image = Image.open(file_)
        st.image(image,clamp=True,use_column_width=True)
        predictions = import_and_predict(image, model)
        result = np.argmax(predictions)
        probablity=np.max(predictions)
        st.success(
            f"Based on provide X-ray ,it seems to be {round(probablity*100,2)} % {class_indices[int(result)]}")
