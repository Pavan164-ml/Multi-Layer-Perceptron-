import tensorflow as tf
def dataset():
    ## Loading Data
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full),(X_test, y_test) = mnist.load_data()

    ## Splitting data into training and validation sets
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


    X_test = X_test / 255.

    return X_valid, X_train,y_valid, y_train,X_test, y_test
