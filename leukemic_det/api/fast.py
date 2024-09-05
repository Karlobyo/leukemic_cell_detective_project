import numpy as np
import tensorflow

from PIL import Image

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

# load model from local directory and cache it
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
    # retrieve testing images never seen by the model
    imgs = show_img_prelim(img_sample)

    # small preproc
    resized = np.resize((imgs), (450, 450, 3))
    X_pred = np.expand_dims(resized, 0)

    # classify
    y_pred = model.predict(X_pred)

    # set threshold
    prediction = (y_pred > 0.5).astype(int)

    # parse the reponse
    if prediction == 0:
        return {"The sample cell is":'Healthy'}
    else:
        return {"The sample cell is":'Malignant'}


@app.post("/classify") # you need a post request when you want to send anything to the server (an image in this case)
async def classify(image: UploadFile=File(...)): # async funcs allow processes to run in parallel, in this case you will be able to have the API endpoint available while waiting for the user to upload the image. As long as the image is not processed the following code won't be executed, thanks to the await keyword

    # decode uploaded image
    image_up = Image.open(image)

    # small preproc
    resized = np.resize((image_up), (450, 450, 3))
    X_pred = np.expand_dims(resized, 0) # add the batch dimension required by CNNs

    # classify
    y_pred = model.predict(X_pred)

    # set threshold
    prediction = (y_pred > 0.5).astype(int)

    # parse the reponse
    if prediction == 0:
        return {"The sample cell is":'Healthy'}
    else:
        return {"The sample cell is":'Malignant'}
