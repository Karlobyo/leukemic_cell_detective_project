import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from params import *
from leukemic_det.ml_logic.data import load_and_preprocess_train_data
from leukemic_det.ml_logic.registry import mlflow_run
from keras.callbacks import EarlyStopping
from tqdm import tqdm


#@mlflow_run
def preprocess_and_train() -> float:
    """
    - Fetch data from GCS and preprocess
    - Train on the preprocessed data 
    - Store model weights

    Return val_accuracy as float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    from leukemic_det.ml_logic.registry import save_model
    from leukemic_det.ml_logic.model import load_base_model
    from leukemic_det.ml_logic.registry import mlflow_transition_model

    print(Fore.BLUE + "\nLoading preprocessed data..." + Style.RESET_ALL)
    
    
    X, y = tqdm(load_and_preprocess_train_data(DATA_SIZE))
    
    split_ratio = 0.2 
    
    train_length = int(len(X)*(1-split_ratio))

    X_train = X[0:train_length]
    X_val = X[train_length:]
    y_train = y[0:train_length]
    y_val = y[train_length:]
    
    
    # Train model using `model.py`
    model = load_base_model()
        
    es = EarlyStopping(patience=10)
    
    history = model.fit(X_train, y_train,
                                 batch_size=64,
                                 callbacks=[es],
                                 validation_data=(X_val, y_val), epochs=25, verbose=0)

    
    val_accuracy = np.max(history.history['val_accuracy'])


    # Save model weight on hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ preprocess_and_train() done \n")
    return val_accuracy
    

#@mlflow_run
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

    X, y = preprocess()

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
       X_pred = preprocess()

    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, "\n")
    return y_pred


if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()