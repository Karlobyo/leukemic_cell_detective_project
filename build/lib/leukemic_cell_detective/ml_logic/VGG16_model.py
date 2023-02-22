from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

def load_model():

    model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)

    return model


model_VGG16 = load_model()


def set_nontrainable_layers(model):

    model.trainable=False

    return model


model_VGG16 = set_nontrainable_layers(model_VGG16)


def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''

    base_model = load_model()

    base_model = set_nontrainable_layers(base_model)

    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')

    model = models.Sequential([
        base_model,
        flattening_layer,
        dense_layer,
        prediction_layer
        ])


    return model


model_VGG16_complete = add_last_layers(model_VGG16)


def build_model():

    model = model_VGG16_complete

    opt = optimizers.Adam(learning_rate=1e-4)

    model.compile(loss='binary_crossentropy', optimizer= opt, metrics='accuracy')

    return model


def train_model():

    es = EarlyStopping(patience=20)
    history = model_ready.fit(X_train, y_train, validation_data=(X_val, y_val),
                          callbacks=[es], batch_size=64, epochs=100)
    return history
