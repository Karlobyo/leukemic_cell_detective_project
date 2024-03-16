import numpy as np
import tensorflow

import cv2 as cv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File

from leukemic_det.ml_logic.data_classification import show_img_prelim

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.state.model = tensorflow.keras.models.load_model("leukemic_det/api/model_dir/20240312-114546.h5")

model = app.state.model

@app.get("/")
def root():
    return dict(greeting="Hello")


@app.get("/predict")
def predict(img_sample : int):
    """
    Make a single image prediction
    Assumes `img_sample' is provided as an integer index by the user
    """

    im = show_img_prelim(img_sample)

    u = np.resize((im), (450, 450, 3))
    resized_u = np.array(u)

    X_pred = np.expand_dims(resized_u, 0)

    y_pred = model.predict(np.array(X_pred))

    predicted_class_u = (y_pred > 0.5).astype(int)

    if predicted_class_u == 0:
        return {"The sample cell is":'Healthy'}
    else:
        return {"The sample cell is":'Malignant'}


# @app.post("/classify")
# async def classify(image: UploadFile=File(...)):

#     file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
#     image_u = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

#     # predict uploaded image

#     u = np.resize((image_u), (450, 450, 3))
#     resized_u = np.array(u)

#     X_pred = np.expand_dims(resized_u, 0)
#     y_pred = model.predict(np.array(X_pred))

#     predicted_class_u = (y_pred > 0.5).astype(int)


#     if predicted_class_u == 0:
#         return {"The sample cell is":'Healthy'}
#     else:
#         return {"The sample cell is":'Malignant'}
