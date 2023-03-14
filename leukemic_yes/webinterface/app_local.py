
import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
import base64
from leukemic_det.ml_logic.data import load_test_img
from leukemic_det.ml_logic.data import load_test_img_prelim
from leukemic_det.ml_logic.data import load_test_data
from leukemic_det.ml_logic.registry import load_model
from fastapi import FastAPI
from google.cloud import storage
import tensorflow


# Set path to your service account credentials file
credentials_path = '/Users/carlobarbini/Documents/Carolingio_LeWagon/service_account_key/le-wagon-1-369318-fb5bec66ff4e.json'

# Create a client object using the credentials file
client = storage.Client.from_service_account_json(credentials_path)
bucket = client.bucket('leukemic-1')


app = FastAPI()
app.state.model = tensorflow.keras.models.load_model(
    "/Users/carlobarbini/code/Karlobyo/leukemic_cell_detective_project/leukemic_det/api/model/cnn_base_simple")


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
def callback():
    st.session_state.button_clicked = True


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

st.title('Leukemic Cell Detective')


st.markdown('### *Detecting healthy vs malignant cells from human white blood cells microscopic images*')

st.markdown('Original dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223')

st.markdown('Dataset publication: Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set] The Cancer Imaging Archive https://doi.org/10.7937/tcia.2019.dc64i46r')

st.markdown('This is a convolution neural network deep learning app meant to deliver real-time predictions classifiying human white blood cells microscopic images as healthy or malignant (acute lymphoblastic leukaemia)')

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
def callback():
    st.session_state.button_clicked = True


image_numbers = [int for int in range(1,1801)]
# create a multi-select widget for selecting image numbers
selected_image_number = st.multiselect('Please select an image (1801 samples available):', image_numbers)


@app.get("/show_img")
def show_img_prelim(img_sample : int):

    test_folder = bucket.blob("C-NMC_Leukemia/testing_data/C-NMC_test_prelim_phase_data")
    test_image_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        image_path = blob.name
        test_image_paths.append(image_path)
    
    
        
    blob = bucket.blob(test_image_paths[img_sample])
    image_bytes = blob.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    test_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    return test_img

for i in selected_image_number:
    st.image(show_img_prelim(i), width=200, caption=f'Human white blood cell #{i}')


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
    
    
    
    predicted_class = predict(i)
    if predicted_class == 0:
        st.write('healthy')
    else:
        st.write('malignant')
    
    