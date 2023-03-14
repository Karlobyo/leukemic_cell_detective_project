
import numpy as np 
import pandas as pd 
import os
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow import keras
from google.cloud import storage
from params import *

# Set path to your service account credentials file
credentials_path = '/Users/carlobarbini/Documents/Carolingio_LeWagon/service_account_key/le-wagon-1-369318-fb5bec66ff4e.json'

# Create a client object using the credentials file
client = storage.Client.from_service_account_json(credentials_path)
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

    for i in [all_0,hem_0]:
        paths = get_path_image(i)
        
        img_data.extend(paths)
        
    
    data = {"img_data":img_data,
        "labels":[np.nan for x in range(len(img_data))]}

    data = pd.DataFrame(data) 
    data.loc[0:2938, "labels"] = 1 # ALL
    data.loc[2938:, "labels"] = 0 # HEM
    
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    data_s = data.iloc[:50, :]
    
    img_list = []
    
    for i in range(len(data_s)):
        
        blob = bucket.blob(data_s['img_data'][i])
        image_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        img_list.append(image)   
    
    
    
    X = np.array(img_list)
    y = np.array(data_s['labels'])
    
    # shuffle the data
    
    return X, y




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



def load_test_img(img_sample: int): # returns unlabelled images from GCS bucket leukemic-1
    
    test_folder = bucket.blob("C-NMC_Leukemia/unlabelled_imgs/C-NMC_test_final_phase_data")
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




def load_test_data(): # returns unlabelled images from GCS bucket leukemic-1
    
    test_folder = bucket.blob("C-NMC_Leukemia/unlabelled_imgs/C-NMC_test_final_phase_data")
    test_image_paths = []
    for blob in bucket.list_blobs(prefix=test_folder.name):
        image_path = blob.name
        test_image_paths.append(image_path)
    
    

    test_img_list = []
    
    for i in range(len(test_image_paths)):
        
        blob = bucket.blob(test_image_paths[i])
        image_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        test_img_list.append(image) 
    
    
    resized_test_imgs = []

    for i in range(len(test_img_list)):
        s = np.resize(test_img_list[i], (450, 450, 3))
        resized_test_imgs.append(s)
    resized_test_imgs = np.array(resized_test_imgs)

    return resized_test_imgs
    
    
    
import concurrent.futures

def process_image(row):
    blob = bucket.blob(row['img_data'])
    image_bytes = blob.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    
def multi_treading_cache():
    img_list_2 = [] # multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_row = {executor.submit(process_image, row): row for _, row in data_s.iterrows()}
        for future in concurrent.futures.as_completed(future_to_row):
            try:
                
                cache = {}

                for i in range(len(data_s)):
                    blob_name = sample_df['img_data'][i]
                    if blob_name in cache:
                        image = cache[blob_name]
                    else:
                        image = future.result()

                    img_list_2.append(image)
                
            except Exception as exc:
                print(f'Error processing image: {exc}')
                
                

def generate_crop_imgs():
    img_list = []
    for i in range(len(img_data)):
        image = cv.imread(data["img_data"][i])
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        result = cv.bitwise_and(image, image, mask=thresh)
        result[thresh==0] = [255,255,255] 
        (x, y, z_) = np.where(result > 0)
        mnx = (np.min(x))
        mxx = (np.max(x))
        mny = (np.min(y))
        mxy = (np.max(y))
        crop_img = image[mnx:mxx,mny:mxy,:]
        crop_img_r = cv.resize(crop_img, (224,224))
        img_list.append(crop_img_r)