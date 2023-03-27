import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from leukemic_det.params import *
from leukemic_det.ml_logic.data import load_and_preprocess_train_data
from leukemic_det.ml_logic.data import load_test_imgs
from leukemic_det.ml_logic.data import load_test_img_prelim
from leukemic_det.ml_logic.registry import mlflow_run
from keras.callbacks import EarlyStopping
from tqdm import tqdm


@mlflow_run
def preprocess_and_train() -> float:
    """
    - Fetch data from GCS and preprocess
    - Train on the preprocessed data 
    - Store model weights

    Return val_accuracy as float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: preprocess and train" + Style.RESET_ALL)
    from leukemic_det.ml_logic.registry import save_model
    from leukemic_det.ml_logic.registry import save_results
    from leukemic_det.ml_logic.model import load_base_model
    from leukemic_det.ml_logic.model import load_VGG16_model
    from leukemic_det.ml_logic.registry import mlflow_transition_model

    print(Fore.BLUE + "\nLoading and preprocessing data..." + Style.RESET_ALL)
    
    
    X, y = tqdm(load_and_preprocess_train_data(DATA_SIZE))
    
    split_ratio = 0.2 
    
    train_length = int(len(X)*(1-split_ratio))

    X_train = X[0:train_length]
    X_val = X[train_length:]
    y_train = y[0:train_length]
    y_val = y[train_length:]
    
    X_s = X_train.shape
    y_s = y_train.shape
    
    print(Fore.CYAN + f"X_train shape is {X_s}" + Style.RESET_ALL)
    print(Fore.CYAN + f"y_train shape is {y_s}" + Style.RESET_ALL)

    
    # Train model 
    if MODEL == "base":
        model = load_base_model()
    elif MODEL == "vgg":
        model = load_VGG16_model()
        
    es = EarlyStopping(patience=5)
    
    print(Fore.BLUE + "\nTraining the model..." + Style.RESET_ALL)
    
    history = model.fit(X_train, y_train,
                                 batch_size=64,
                                 callbacks=[es],
                                 validation_data=(X_val, y_val), epochs=25, verbose=1)

    
    val_accuracy = history.history['val_accuracy'][-1]

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train),
    )

    # Save model weight on hard drive (and optionally on GCS too!)
    save_results(params=params, metrics=dict(val_accuracy=val_accuracy))
    save_model(model=model)

    # The latest model should be moved to staging
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ preprocess_and_train() done \n")
    print(f"The model has an accuracy of {val_accuracy}")
    return val_accuracy
    

@mlflow_run
def evaluate(stage: str = "Production") -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return accuracy as float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)
    from leukemic_det.ml_logic.model import evaluate_model
    from leukemic_det.ml_logic.registry import load_model, save_results

    model = load_model(stage=stage)
    assert model is not None

    X, y = load_test_imgs()

    metrics_dict = evaluate_model(model=model, X=X, y=y)
    accuracy = metrics_dict["accuracy"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")
    
    return accuracy


def pred(X_pred: np.ndarray = None) -> int:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    from leukemic_det.ml_logic.registry import load_model
    
    if X_pred is None:
       X_pred = load_test_img_prelim(np.random.randint(low=0, high=1000))

    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, "\n")
    return y_pred


if __name__ == '__main__':
    preprocess_and_train()
    evaluate()
    pred()