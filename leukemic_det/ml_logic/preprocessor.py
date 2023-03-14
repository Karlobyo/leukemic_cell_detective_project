from tensorflow import keras
from keras.applications.vgg16 import preprocess_input
import cv2 as cv

def preprocess_data_VGG16(X):
    
    X_prep = preprocess_input(X)

    return X_prep


#def preprocess_data_base(X):
    





def preprocess_data_base_crop(X):
    
    image = cv.imread(X)
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
    
    X_preprocessed = crop_img_r
    
    return X_preprocessed

