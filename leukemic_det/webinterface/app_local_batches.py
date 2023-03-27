
import cv2 as cv
import streamlit as st
import numpy as np
import base64
#from leukemic_det.ml_logic.data import load_test_img_prelim
from fastapi import FastAPI
from google.cloud import storage
import tensorflow
from leukemic_det.params import *

BATCH_SIZE = int(BATCH_SIZE)

# Create a client object using the credentials file
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


app = FastAPI()
app.state.model = tensorflow.keras.models.load_model(
            '/Users/carlobarbini/code/Karlobyo/leukemic_cell_detective_project/leukemic_det/webinterface/cnn_base_simple')


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

st.title('Leukemic Cell Detective')


st.markdown('### *Detecting healthy vs malignant cells from human white blood cells microscopic images*')

st.markdown('')

st.markdown('***')

st.markdown('')

st.markdown('Original dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223')

st.markdown('Dataset publication: Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set] The Cancer Imaging Archive https://doi.org/10.7937/tcia.2019.dc64i46r')

st.markdown('')

st.markdown('This is a convolution neural network deep learning app meant to deliver real-time predictions classifiying human white blood cells microscopic images as healthy or malignant (acute lymphoblastic leukaemia)')

st.markdown('')

st.markdown('***')

st.markdown('')


def load_test_img_prelim(img_sample: int): # returns unlabelled images from GCS bucket leukemic-1
    
    test_folder = bucket.blob("C-NMC_Leukemia/testing_data/C-NMC_test_prelim_phase_data")
    test_image_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        image_path = blob.name
        test_image_paths.append(image_path)
    
    
        
    blob = bucket.blob(test_image_paths[img_sample])
    image_bytes = blob.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    test_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    

    s = np.resize((test_img), (450, 450, 3))
    resized_test_img = np.array(s)

    return resized_test_img



@app.get("/show_img")
def show_img_prelim(batch : int):

    test_folder = bucket.blob("C-NMC_Leukemia/testing_data/C-NMC_test_prelim_phase_data")
    test_image_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        image_path = blob.name
        test_image_paths.append(image_path)
    
    test_imgs =[]
    
    for i in range((batch-1)*BATCH_SIZE, BATCH_SIZE*batch):
    
        blob = bucket.blob(test_image_paths[i])
        image_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        test_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        test_imgs.append(test_img)
    
    return test_imgs


@app.get("/predict")
def predict(img_sample : int):
    """
    Make a single image prediction
    Assumes `img_sample' is provided as an integer index by the user
    """
    
    X_pred = load_test_img_prelim(img_sample)
    
    model = app.state.model
    assert model is not None
    
    X_pred = np.expand_dims(X_pred, 0)   
    y_pred = model.predict(np.array(X_pred))
    
    y_pred = (y_pred > 0.5).astype(int)
    
    return y_pred


n_batches = int(round(1800 / BATCH_SIZE))
batches = list(range(n_batches))
selected_batch_number = st.multiselect(f'Please select an image batch ({n_batches} available):', batches)

if selected_batch_number:
    i = selected_batch_number[-1]
    img_list = show_img_prelim(i)

    captions = []
    for idx, img in enumerate(img_list):
        idx = idx + 1
        captions.append(f'Human white blood cell #{idx}')
    st.image(img_list, width=200, caption=captions)

    img_number = [k for k in list(range(1, BATCH_SIZE+1))]
    selected_img_number = st.multiselect('Please select an image to be classified from the selected batch:', img_number)

    if selected_img_number:
        j = selected_img_number[-1]
        j=j-1
        im = show_img_prelim(i)[j]
        st.image(im, width=200, caption=f'Human white blood cell #{j}')
         
        predicted_class = predict(selected_img_number[0]+((i-1)*BATCH_SIZE))
    
        if predicted_class == 0:
            st.write('Healthy')
        else:
            st.write('Malignant')
    