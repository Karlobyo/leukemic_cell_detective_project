import numpy as np
import cv2 as cv
import tensorflow as tf
from google.cloud import storage
import streamlit as st
from google.oauth2 import service_account


# Create a client object using the credentials file
service_account_info = st.secrets["gcp_service_account"]

# Create credentials object from the secrets
credentials = service_account.Credentials.from_service_account_info(service_account_info)
#credentials = service_account_info["private_key"]

#credentials, project = auth.default()

bucket = st.secrets["bucket"]

# Initialize the client with the credentials

client = storage.Client(project=service_account_info["project_id"], credentials=credentials)

bucket = client.bucket(bucket)

model = tf.keras.models.load_model(
    "leukemic_det/webinterface/model_dir/20240312-114546.h5")

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


def predict(X_pred):
    """
    Makes a single image prediction
    Returns 0 for healthy and 1 for malignant
    """

    resized = np.resize((X_pred), (450, 450, 3))

    X_pred = np.expand_dims(resized, 0)

    y_pred = model.predict(np.array(X_pred))

    predicted_class = (y_pred > 0.5).astype(int)

    return predicted_class
