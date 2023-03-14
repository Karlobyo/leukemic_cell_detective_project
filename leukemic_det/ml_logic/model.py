
from tensorflow import keras
from keras import layers, models
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np


def load_base_model():

    model = models.Sequential()
    
    model.add(layers.Rescaling(1./450, input_shape=(450,450,3)))

    model.add(layers.Conv2D(8, (5, 5), input_shape = (450, 450, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Conv2D(16, (4, 4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    #model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(layers.MaxPool2D(pool_size=(2,2)))
    #model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    #model.add(layers.Dense(100))

    model.add(layers.Dense(50))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])

    return model



def evaluate_model(model: models,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> tuple[models, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ model evaluated: mae {round(mae, 2)}")

    return metrics