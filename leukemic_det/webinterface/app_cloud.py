import streamlit as st

import base64
import cv2 as cv

import numpy as np
import tensorflow as tf

from google.cloud import storage



# Retrieve the secrets
service_account_info = st.secrets["gcp_service_account"]

# Create credentials object from the secrets
credentials = service_account.Credentials.from_service_account_info(service_account_info)
#credentials = service_account_info["private_key"]

#credentials, project = auth.default()

bucket = st.secrets["bucket"]

# Initialize the client with the credentials

client = storage.Client(project=service_account_info["project_id"], credentials=credentials)

bucket = client.bucket(bucket)


st.set_page_config(layout='wide')


# model for classifying uploaded image (when there is no active Google Cloud Run URL)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
    "leukemic_det/webinterface/model_dir/20240312-114546.h5")

model = load_model()

### functions ###


# getting paths of test images
@st.cache_data
def get_imgs_paths():
    test_folder = bucket.blob("C-NMC_Leukemia/testing_data/C-NMC_test_prelim_phase_data")
    imgs_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        blob_image_paths = blob.name
        imgs_paths.append(blob_image_paths)
    return imgs_paths


# display chosen image
@st.cache_data(show_spinner=False)
def show_img(img_sample : int):

    imgs_paths = get_imgs_paths()

    # decoding the img path
    img_path = bucket.blob(imgs_paths[img_sample])
    image_bytes = img_path.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    chosen_img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    return chosen_img


# classify image
def predict(X_pred):
    """
    Makes a single image prediction
    Returns 0 for healthy and 1 for malignant
    """
    resized = np.resize((X_pred), (450, 450, 3))
    X_pred = np.expand_dims(resized, 0)
    y_pred = model.predict(X_pred)
    predicted_class = (y_pred > 0.5).astype(int)
    return predicted_class



# add bg image
@st.cache_data
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


st.markdown('### *Detecting malignant vs healthy human white blood cells from microscopic images*')

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
