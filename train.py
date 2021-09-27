from utils import data
from utils.model import mlp
from utils.data import *
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging


def main():
    
    logging.basicConfig(level=logging.INFO)
    logging.info("-------------------------------------------------------------------")
    logging.info("Separating the data set into Training ,Validation and Testing sets")
    (X_valid, X_train,y_valid, y_train,X_test, y_test) = dataset()
    
    logging.info("Creating the model using Modelling function")

    LAYERS = mlp.modelling()


    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]


    model_clf = tf.keras.models.Sequential(LAYERS)

    logging.info("Compiling the model")
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    EPOCHS = 30
    epochs = EPOCHS
    VALIDATION = (X_valid, y_valid)
    # Just kiddings of
    history = model_clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)
    print(history.history)
    logging.info("-------------------------------------------------------------------")
        

if __name__ == "__main__":
    main()