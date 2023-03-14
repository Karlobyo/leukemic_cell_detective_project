import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from leukemic_det.ml_logic.data import load_test_img
from leukemic_det.ml_logic.registry import load_model
import tensorflow
import matplotlib.pyplot as plt
import cv2 as cv


app = FastAPI()

app.state.model = tensorflow.keras.models.load_model("leukemic_det/api/model/cnn_base_simple")


@app.get("/show_img")
def show_img(img_sample : int):
    """
    Displays image to be classified.
    Assumes `img_sample' is provided as an integer index by the user
    """
    
    img = load_test_img(img_sample)
    
    img = cv.imdecode(img, cv.IMREAD_COLOR)
        
    return img




@app.get("/predict")
def predict(img_sample : int):
    """
    Make a single image prediction
    Assumes `img_sample' is provided as an integer index by the user
    """
    
    X_pred = load_test_img(img_sample)
        
    model = app.state.model
    
    assert model is not None
    
    X_pred = np.expand_dims(X_pred, 0)   
    y_pred = model.predict(X_pred)
    
    y_pred = (y_pred > 0.5).astype(int)
    
    return {'classification': float(y_pred[0])}



@app.get("/")
def root():
    return {'greeting':'gianni'}