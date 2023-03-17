
import cv2 as cv
import streamlit as st
import numpy as np
import base64
from fastapi import FastAPI
from google.cloud import storage
import tensorflow

# Create a client object using the credentials file
client = storage.Client()
bucket = client.bucket('leukemic-1')

app = FastAPI()
app.state.model = tensorflow.keras.models.load_model(
            'leukemic_det/webinterface/cnn_base_simple')
model = app.state.model

# if 'button_clicked' not in st.session_state:
#     st.session_state.button_clicked = False
# def callback():
#     st.session_state.button_clicked = True


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

add_bg_from_local('images/lympho.png')

st.title('Leukemic Cell Detective')


st.markdown('### *Detecting healthy vs malignant cells from human white blood cells microscopic images*')

st.markdown('')

st.markdown('***')

st.markdown('')


st.markdown('Original dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223')

st.markdown('Dataset publication: Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set] The Cancer Imaging Archive https://doi.org/10.7937/tcia.2019.dc64i46r')

st.markdown('')

st.markdown('This is a research preview of a convolution neural network deep learning app meant to deliver real-time predictions classifiying human white blood cells  \nmicroscopic images as healthy or malignant (acute lymphoblastic leukaemia)')

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
def show_img_prelim(img_sample : int):

    test_folder = bucket.blob("C-NMC_Leukemia/testing_data/C-NMC_test_prelim_phase_data")
    test_image_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        image_path = blob.name
        test_image_paths.append(image_path)
    
    test_imgs =[]
    
    
    blob = bucket.blob(test_image_paths[img_sample])
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
    
    assert model is not None
    
    X_pred = np.expand_dims(X_pred, 0)   
    y_pred = model.predict(np.array(X_pred))
    
    y_pred = (y_pred > 0.5).astype(int)
    
    return y_pred



st.markdown('Please select an image to be classified (1800 available):')

img_number = [k for k in list(range(1, 1801))]
selected_img_number = st.multiselect('', img_number)

if selected_img_number:
    j = selected_img_number[-1]
    j=j-1
    im = show_img_prelim(j)
    # im = np.resize(im, (100, 100, 3))
    st.image(im, width=200, caption=f'Human white blood cell #{j+1}')

    
    #if (st.button('Classify',on_click=callback) or st.session_state.button_clicked):
        
    predicted_class = predict(selected_img_number[-1])

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
    
    
    # predict
    
    #if (st.button('Classify uploaded image',on_click=callback, key='upload') or st.session_state.button_clicked):

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
    
st.markdown('-In the best case your image should show an individual cell well defined from a black background-')
