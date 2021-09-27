import logging
from tqdm import tqdm
import tensorflow as tf
class mlp:
    """
    This class is used to implement Multilayer perceptron

    """
    def modelling():
        LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")]
          
        return LAYERS   
