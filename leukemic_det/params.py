import os
import numpy as np
from google.cloud import storage


##################  VARIABLES  ##################

URL=os.environ.get("URL")
BATCH_SIZE=os.environ.get("BATCH_SIZE")
DATA_SIZE = int(os.environ.get("DATA_SIZE"))
CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MODEL=os.environ.get("MODEL")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

GCR_IMAGE = os.environ.get("GCR_IMAGE")
GCR_REGION = os.environ.get("GCR_REGION")
GCR_MEMORY = os.environ.get("GCR_MEMORY")

LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".leukemic", "models")
