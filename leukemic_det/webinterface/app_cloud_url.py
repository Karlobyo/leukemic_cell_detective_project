
import streamlit as st

import base64
import cv2 as cv

import numpy as np
import tensorflow as tf

import requests

from google.cloud import storage


# Create a client object
client = storage.Client()
bucket = client.bucket("leukemic-1")

# model for classifying uploaded image (when there is no active Google Cloud Run URL)
model = tf.keras.models.load_model(
    "leukemic_det/webinterface/model_dir/20240312-114546.h5")


URL = ""


# display image
def show_img_prelim(img_sample : int):

    # getting bucket paths of test images
    test_folder = bucket.blob("C-NMC_Leukemia/testing_data/C-NMC_test_prelim_phase_data")
    test_image_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        image_path = blob.name
        test_image_paths.append(image_path)

    # deconding the imgs paths into images
    test_imgs =[]
    blob = bucket.blob(test_image_paths[img_sample])
    image_bytes = blob.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    test_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    test_imgs.append(test_img)

    return test_imgs


# add bg image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


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


st.markdown('### *Detecting healthy vs malignant cells from human white blood cells microscopic images*')

st.markdown('')

st.markdown('***')

st.markdown('')


st.markdown('Original dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223')

st.markdown('Dataset publication: Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]   \nThe Cancer Imaging Archive https://doi.org/10.7937/tcia.2019.dc64i46r')

st.markdown('')

st.markdown('This is a research preview of a convolution neural network deep learning app meant to deliver real-time predictions classifiying   \nhuman white blood cells microscopic images as healthy or malignant (acute lymphoblastic leukaemia)')

st.markdown('***')

st.markdown('')


# create multiselect widget for choosing an image
st.markdown('Please select an image to be classified (1800 available):')

img_number = [k for k in list(range(1, 1801))]
selected_img_number = st.multiselect('', img_number)



if selected_img_number:
    j = selected_img_number[-1]
    j=j-1
    im = show_img_prelim(j)
    st.image(im, width=200, caption=f'Human white blood cell #{j+1}')

    # predict chosen image

    # use api
    leukemic_api_url = f'{URL}/predict'
    params = {'img_sample':selected_img_number[-1]}
    response = requests.get(leukemic_api_url, params=params)

    prediction = response.json()

    predicted_class = prediction['The sample cell is']


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
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_u = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    st.image(image_u, width=200, channels="BGR", caption='uploaded image')


    # predict uploaded image

    u = np.resize((image_u), (450, 450, 3))
    resized_u = np.array(u)

    X_pred = np.expand_dims(resized_u, 0)
    y_pred = model.predict(np.array(X_pred))

    predicted_class_u = (y_pred > 0.5).astype(int)

    if predicted_class_u == 0:
        st.write('Healthy')
    else:
        st.write('Malignant')


st.markdown('')

st.markdown('')

st.markdown('-The model works best if your image shows an individual white blood cell well defined from a black background-')
