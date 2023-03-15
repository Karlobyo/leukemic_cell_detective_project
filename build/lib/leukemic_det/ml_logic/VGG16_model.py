from tensorflow import keras
from keras import layers, models
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.callbacks import EarlyStopping

def load_VGG16_model():

    VGG_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)
    VGG_model.trainable=False

    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')

    model = models.Sequential([
        VGG_model,
        flattening_layer,
        dense_layer,
        prediction_layer
        ])


    opt = optimizers.Adam(learning_rate=1e-4)

    model.compile(loss='binary_crossentropy', optimizer= opt, metrics='accuracy')

    return model







def train_model(model):

    es = EarlyStopping(patience=20)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          callbacks=[es], batch_size=64, epochs=100)
    return history



def load_pre_trained_model():

    model = keras.models.load_model(path)

    return model
