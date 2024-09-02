import tensorflow as tf
import streamlit as st

from google.cloud import storage
from google.oauth2 import service_account

from leukemic_det.params import *


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads model from local directly or from gcs
    """
    if MODEL_TARGET == "local":

        return tf.keras.models.load_model(
        "leukemic_det/webinterface/model_dir/20240312-114546.h5")


    elif MODEL_TARGET == "gcs":

        if UI == "local":

            client = storage.Client()
            #breakpoint()
            bucket = client.bucket(BUCKET_NAME)

            blobs = list(client.get_bucket(bucket).list_blobs(prefix="models"))

            try:
                latest_blob = max(blobs, key=lambda x: x.updated)

                latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name).replace("models/", "")
                #breakpoint()
                latest_blob.download_to_filename(latest_model_path_to_save)

                latest_model = tf.keras.models.load_model(latest_model_path_to_save)

                print("✅ Latest model downloaded from cloud storage")

                return latest_model
            except:
                print(f"\n❌ No model found in GCS bucket {bucket}")

                return None

        elif UI == "cloud":

            # Retrieve the gcp account secrets
            service_account_info = st.secrets["gcp_service_account"]

            # Create credentials and bucket object from the secrets
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            bucket = st.secrets["bucket"]

            # Initialize the client with the credentials
            client = storage.Client(project=service_account_info["project_id"], credentials=credentials)
            bucket = client.bucket(bucket)

            blobs = list(client.get_bucket(bucket).list_blobs(prefix="models"))

            try:
                latest_blob = max(blobs, key=lambda x: x.updated)
                latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name).replace("models/", "")
                latest_blob.download_to_filename(latest_model_path_to_save)

                latest_model = tf.keras.models.load_model(latest_model_path_to_save)

                print("✅ Latest model downloaded from cloud storage")

                return latest_model
            except:
                print(f"\n❌ No model found in GCS bucket {bucket}")

                return None



if __name__=='__main__':
    load_model()
