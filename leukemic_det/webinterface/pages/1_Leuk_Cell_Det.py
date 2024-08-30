import streamlit as st

import cv2 as cv

import numpy as np

from google.cloud import storage
from google.oauth2 import service_account

from bg_loader import add_bg_from_local
from leukemic_det.ml_logic.data_classification import show_img, predict

# Retrieve the gcp account secrets
service_account_info = st.secrets["gcp_service_account"]

# Create credentials and bucket object from the secrets
credentials = service_account.Credentials.from_service_account_info(service_account_info)
bucket = st.secrets["bucket"]

# Initialize the client with the credentials
client = storage.Client(project=service_account_info["project_id"], credentials=credentials)
bucket = client.bucket(bucket)


st.set_page_config(layout='wide')


CSS = """
h1 {
    color: white;
}
h2 {color: black;
}
.stApp {
    background-image: webinterface/app_cloud.py;
    background-size: cover;
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

add_bg_from_local('leukemic_det/webinterface/images/lympho.png')

st.title('Leukemic Cell Detective')


st.markdown("""### *Detecting healthy vs malignant cells*\n ### *from human white blood cells microscopic images*""")

st.markdown('')

st.markdown('***')

st.markdown('')


# create multiselect widget for choosing an image
st.markdown('Please select an image to be classified (1800 available):')

img_number = [k for k in list(range(1, 1801))]
selected_img_number = st.selectbox('', img_number, index=None)


if selected_img_number:
    img = show_img(selected_img_number)

    # display selected image
    st.image(img, width=200, caption=f'Human white blood cell #{selected_img_number}')

    # classify selected image
    predicted_class = predict(img)

    if predicted_class == 0:
        st.write('Healthy')
    else:
        st.write('Malignant')


# image uploader
st.markdown('')

st.markdown('***')

st.markdown('')

st.markdown('Or upload an image from your browser:')

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:

    # decode
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_u = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    # display uploaded image
    st.image(image_u, width=200, channels="BGR", caption='uploaded image')

    # classify uploaded image
    predicted_class_u = predict(image_u)

    if predicted_class_u == 0:
        st.write('Healthy')
    else:
        st.write('Malignant')


st.markdown('')

st.markdown('')

st.markdown('-The model works best if your image shows an individual white blood cell well defined from a black background-')
