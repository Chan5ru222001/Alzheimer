import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_lottie import st_lottie
import requests
from PIL import Image, ImageOps



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
    model = tf.keras.models.load_model(
        "pages/../Alzheimer.h5")
    return model


model = model_()
col1, col2, col3 = st.columns([3 ,11, 3])

class_indices = {0: 'Mild Dementia', 1: 'Moderate Dementia',
                 2: 'Non Demented', 3: 'Very mild Dementia'}

def import_and_predict(image_data, model):
    size = (299, 299)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)

    return prediction

with col2:
    st.title("Discover Your Outcome")
    file_ = st.file_uploader("Please upload the Brain X-ray image", type=['jpg'])
    st.markdown("""___""")


if file_ is None:
    st.text(" ")
else:
    with col2:
        image = Image.open(file_)
        st.image(image,clamp=True,use_column_width=True)
        predictions = import_and_predict(image, model)
        result = np.argmax(predictions, axis=1)
        st.success(
            f"Based on provide X-ray ,it seems to be {class_indices[int(result)]}")
