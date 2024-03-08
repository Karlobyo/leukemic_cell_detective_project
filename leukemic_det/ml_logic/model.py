
from keras import layers, models
import numpy as np
from colorama import Fore, Style
from keras.applications.vgg16 import VGG16
from keras import optimizers

# model architecture
def load_base_model():

    model = models.Sequential()

    model.add(layers.Rescaling(1./450, input_shape=(450,450,3)))

    model.add(layers.Conv2D(8, (5, 5), input_shape = (450, 450, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Conv2D(16, (4, 4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(10))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# VGG16 Transfer Learning model
def load_VGG16_model():

    VGG_model = VGG16(weights="imagenet", include_top=False, input_shape=(450, 450, 3))
    VGG_model.trainable=False

    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(10, activation='relu')
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

# model evaluation
def evaluate_model(model: models,
                   X: np.ndarray,
                   y: np.ndarray):
    """
    Evaluate trained model performance on test data
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        verbose=0,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ model evaluated: accuracy {round(accuracy, 2)}")

    return metrics
