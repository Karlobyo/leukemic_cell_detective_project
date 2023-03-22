from tensorflow import keras
from keras import layers, models
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.callbacks import EarlyStopping

def load_VGG16_model():

    VGG_model = VGG16(weights="imagenet", include_top=False, input_shape=(450, 450, 3))
    VGG_model.trainable=False

    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(50, activation='relu')
    prediction_layer = layers.Dense(1, activation='sigmoid')

    model = models.Sequential([
        VGG_model,
        flattening_layer,
        dense_layer,
        prediction_layer
        ])


    opt = optimizers.Adam(learning_rate=1e-4)

    model.compile(loss='binary_crossentropy', optimizer= opt, metrics='accuracy')

    return model


