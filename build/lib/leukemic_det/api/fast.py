from fastapi import FastAPI
import numpy as np
from leukemic_det.ml_logic.data import load_test_img_prelim
import tensorflow
import cv2 as cv


app = FastAPI()

app.state.model = tensorflow.keras.models.load_model("cnn_base_simple")
model = app.state.model

@app.get("/")
def root():
    return {'greeting':'hello'}


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

