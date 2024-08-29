
import streamlit as st

#import tensorflow

import requests

from google.cloud import storage
from google.oauth2 import service_account


# functions
from leukemic_det.ml_logic.data_classification import show_img_prelim
from leukemic_det.webinterface.bg_loader import add_bg_from_local



URL = "http://127.0.0.1:8000"


# Create a client object using the credentials file
# Retrieve the secrets
#service_account_info = st.secrets["gcp_service_account"]

# Create credentials object from the secrets
#credentials = service_account.Credentials.from_service_account_info(service_account_info)
#credentials = service_account_info["private_key"]

#credentials, project = auth.default()

bucket = st.secrets["bucket"] # this comes from the toml file in the root directory

# Initialize the client with the credentials

client = storage.Client()#project=service_account_info["project_id"], credentials=credentials)

bucket = client.bucket(bucket)
#model = tensorflow.keras.models.load_model(
#    "leukemic_det/webinterface/model_dir/20240312-114546.h5")



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


add_bg_from_local('/Users/carlobarbini/code/Karlobyo/leukemic_cell_det_project/leukemic_cell_detective_project/leukemic_det/webinterface/images/lympho.png')


st.title('Leukemic Cell Detective')


st.markdown('### *Detecting healthy vs malignant cells from human white blood cells microscopic images*')

st.markdown('')

st.markdown('')


st.markdown('***')

st.markdown('')


# create multiselect widget for choosing an image

st.markdown('Please select an image to be classified (1800 available):')

img_number = [k for k in list(range(1, 1801))]
selected_img_number = st.multiselect('', img_number)



if selected_img_number:
    img_index = selected_img_number[-1]
    img = show_img_prelim(img_index)
    st.image(img, width=200, caption=f'Human white blood cell #{img_index}')

    # predict chosen image

    # use api
    predict_api_url = f'{URL}/predict'
    params = {'img_sample':selected_img_number[-1]}
    response = requests.get(predict_api_url, params=params)

    prediction = response.json()

    predicted_class = prediction['The sample cell is']

    #predicted_class = predict(img)

    if predicted_class == 0:
        st.write('Healthy')
    else:
        st.write('Malignant')




# image uploader
st.markdown('')

st.markdown('***')

st.markdown('')

st.markdown('Or upload an image from your browser:')


st.markdown('')

st.markdown('')



# use post api
classify_api_url = f'{URL}/classify'

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=240)

    # Send the image to the FastAPI endpoint
    files = {"image": uploaded_file.getvalue()}  # `uploaded_file.getvalue()` returns bytes

    try:
        response = requests.post(classify_api_url, files=files)

        if response.status_code == 200:

            prediction = response.json()

            predicted_class = prediction['The sample cell is']

            #predicted_class = predict(img)

            if predicted_class == 0:
                st.write('Healthy')
            else:
                st.write('Malignant')

        else:
            st.write(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.write(f"Request failed: {e}")


st.markdown('')

st.markdown('')

st.markdown('')


st.markdown('-The model works best if your image shows an individual white blood cell well defined from a black background-')
