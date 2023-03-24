
import numpy as np 
import pandas as pd 
import cv2 as cv
from google.cloud import storage
from params import *

# Set path to your service account credentials file
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def get_path_image(folder):
    image_paths = []
    for blob in bucket.list_blobs(prefix=folder.name):
        image_path = blob.name
        image_paths.append(image_path)
    return image_paths


def load_and_preprocess_train_data():
    
    # returns imaging data as X and labels as y both as numpy arrays
    
    
    all_0 = bucket.blob("C-NMC_Leukemia/training_data/fold_0/all")
    all_1 = bucket.blob("C-NMC_Leukemia/training_data/fold_1/all")
    all_2 = bucket.blob("C-NMC_Leukemia/training_data/fold_2/all")

    hem_0 = bucket.blob("C-NMC_Leukemia/training_data/fold_0/hem")
    hem_1 = bucket.blob("C-NMC_Leukemia/training_data/fold_1/hem")
    hem_2 = bucket.blob("C-NMC_Leukemia/training_data/fold_2/hem")
    
    img_data = []

    for i in [all_0, all_1, all_2, hem_0, hem_1, hem_2]:
        paths = get_path_image(i)
        
        img_data.extend(paths)
        
    
    data = {"img_data":img_data,
        "labels":[np.nan for x in range(len(img_data))]}

    data = pd.DataFrame(data) 
    data.loc[0:7272, "labels"] = 1 # ALL
    data.loc[7272:, "labels"] = 0 # HEM
    
    # shuffle the data
    
    data = data.sample(frac=1).reset_index(drop=True)
        
    img_list = []
    
    for i in range(len(data)):
        
        blob = bucket.blob(data['img_data'][i])
        image_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        img_list.append(image)   
    
    
    
    X = np.array(img_list)
    y = np.array(data['labels'])
    
    
    return X, y

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
