# Imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, TimeDistributed, Lambda, \
    MaxPooling2D
from keras.layers import LSTM
from keras import callbacks
import tensorflow.keras.backend as K
import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from extractCountyData import truth_data_distribution

class VanillaModel():
    def __init__(self):
        self.width = 10000
        self.height = 10000
        self.numChannels = 7
        self.inputShape = (self.width, self.height, self.numChannels)
        self.model = None
        self.history = None
        self.data_generator = None #TODO Add here

    def define_init(self):
        #Sequential neural net model definition

    def compile(self):
        #Compile model with given optimizer, hyperparameters

    def train(self):
        #Train model and store in model parameters
        self.history = None #TODO


"""
Function: Split into train/dev/test and perform pre-procesing of data
@Param: X np.ndarray of shape (# samples, # channels, H, W), y np.ndarray of shape (# samples, 1)
@Return: X train, X val, X test, y train, y val, y test
"""
def process_data(X, y):

    # Split Raw Data
    X_train, X_val, X_test, y_train, y_val, y_test = split_raw_data(X, y, train_size = 0.8, val_size = 0.1, test_size = 0.1)

    # Perform Post-Processing (normalization etc.)
    X_train = X_train / 255.
    X_val = X_val / 255.
    X_test = X_test / 255.
    # TODO: Add more post-processing

    return X_train, X_val, X_test, y_train, y_val, y_test

"""
Function: This function splits raw data into train/val/test
@Param: X np.ndarray, y np.ndarray, train size, val size, test size
@Return: X_train, X_val, X_test, y_train, y_val, y_test
"""
def split_raw_data(X, y, train_size = 0.8, val_size = 0.1, test_size = 0.1):

    # Check percentage split
    if train_size + val_size + test_size != 1:
        raise Exception("Train/Dev/Test split percentages must add to 1")

    # compute number of samples to split
    num_samples = X.shape[0]
    num_train = math.floor(num_samples * train_size)
    num_val = math.floor(num_samples * val_size)
    num_test = math.floor(num_samples * test_size)

    # Perform split
    X_train = X[:num_train]
    X_val = X[num_train: num_train + num_val]
    X_test = X[-num_test:]
    y_train = y[:num_train]
    y_val = y[num_train: num_train + num_val]
    y_test = y[-num_test:]

    return X_train, X_val, X_test, y_train, y_val, y_test

"""
Function: Function that defines the NN architecture (based on reference paper)
@Params: None
@Return: Keras sequential model
Note: Reference architecture for CNN from https://github.com/brad-ross/crop-yield-prediction-project/tree/master/model
"""
def NN_model_definition():

    # Define sequential model
    model = keras.models.Sequential()

    # CNN Definition
    model.add(Conv2D(128, kernel_size=3, padding="same", activation="relu", strides=1, input_shape=(7, 5000, 5000))) #Todo: Put in input shape
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(128, kernel_size=3, padding="same", activation="relu", strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Conv2D(256, kernel_size=3, padding="same", activation="relu", strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(256, kernel_size=3, padding="same", activation="relu", strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Conv2D(512, kernel_size=3, padding="same", activation="relu", strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(512, kernel_size=3, padding="same", activation="relu", strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(512, kernel_size=3, padding="same", activation="relu", strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048))  # Todo: Fix last layer

    # output model summary
    model.summary()

    # Compile model
    selected_optimizer = optimizers.Adam(lr=0.001,
                                         beta_1=0.9,
                                         beta_2=0.999)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=selected_optimizer,
                  metrics=["accuracy"])

    return model

"""
Function: Function that contains all the options/config for model training
@Params: NN model Keras Object, X Train, y train, X val, y val
@Return: History object of trained model
"""
def train_model(NN_model, X_train, y_train, X_valid, y_valid):

    # Define model callbacks
    checkpoint_cb = callbacks.ModelCheckpoint("sandbox_model.h5") #TODO: Model naming convention
    early_stopping_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    run_logdir = get_run_logdir()
    tensorboard_cb = callbacks.TensorBoard(run_logdir)

    # Fit model
    history = NN_model.fit(X_train,
                           y_train,
                           batch_size = 64,
                           epochs = 5,
                           shuffle = True,
                           validation_data = (X_valid, y_valid),
                           callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    return history

"""
Helper Function: This function creates a path to the store the logs for the NN model training run
@Param: none
@Return: specified path to save logs
"""
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    root_logdir = os.path.join(os.curdir, "my_logs")
    return os.path.join(root_logdir, run_id)

"""
Function: Placeholder function for all the data analyses that will take place.
Note: For now, the function includes a plot for lost of validation set
@Params: History object from trained model
@Return: None
"""
def analyze_data(history):

    # Added bogus plot for validation lost
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

if __name__ == "__main__":


    """
    # run logistics
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    bin_array = truth_data_distribution(filename ="../json_store/Illinois_Soybeans_Truth_Data.csv", num_classes = 10)

    # Load Raw Data
    X, y = np.load('training_set_with_labels.npy')

    # apply pre-processing and split data-set
    X_train, X_valid, X_test, y_train, y_valid, y_test = process_data(X, y)

    # Define and Compile model
    NN_model = NN_model_definition()
    
    # Train Model
    history = train_model(NN_model, X_train, y_train, X_valid, y_valid)
    """

    """
    # Plot Metrics
    analyze_data(history)
    
    # Evaluate Model on Test Set
    # NN_model = keras.models.load_model("TBD_name_model.h5") # load in previous model to evaluate
    NN_model.evaluate(X_test, y_test)"""















"""
Archived Functions =====================================================================
"""

"""
Function: This function splits the training set into the desired train/validation split
@Param: X train, y train, desired percentage for training set
@Return: X train, X val, y train, y val
"""
def split_train_data(X_train_full, y_train_full, train_percent=0.9):
    # Compute number of samples to split up
    num_samples = X_train_full.shape[0]
    num_val_samples = math.floor(1 - train_percent * num_samples)

    # perform split
    X_train, X_valid = X_train_full[:-num_val_samples], X_train_full[-num_val_samples:]
    y_train, y_valid = y_train_full[:-num_val_samples], y_train_full[-num_val_samples:]

    return X_train, X_valid, y_train, y_valid


"""
Sample Function: Perform pre-procesing of data and split train into train/dev for CIFAR10
@Param: Raw data of form ((x train, y train), (x test, y test))
@Return: X train, X val, X test, y train, y val, y test
X train shape: (num samples, H, W, channels)
Y train shape: (num samples, 1)
"""
def process_sample_data(acquired_data):
    # Isolate tuples
    full_training = acquired_data[0]
    full_test = acquired_data[1]
    X_training = full_training[0]
    y_training = full_training[1]
    X_test = full_test[0]
    y_test = full_test[1]

    # Perform Pre-processing (cloud removal, normalization, etc.)
    X_training = X_training / 255.
    X_test = X_test / 255.

    # Split training data into train/validation
    X_train, X_valid, y_train, y_valid = split_train_data(X_training, y_training)

    # Reshape arrays
    """X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    y_valid = y_valid[..., np.newaxis]
    y_test = y_test[..., np.newaxis]"""

    return X_train, X_valid, X_test, y_train, y_valid, y_test

"""
Test Function: This test function verifies the workflow of the various function in this file on a CIFAR10.
@Params: None
@Return: None
"""
def test_workflow_CIFAR10():

    # run logistics
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load Raw Data TODONOW
    raw_data = tf.keras.datasets.cifar10.load_data()

    # apply pre-processing and split data-set
    X_train, X_valid, X_test, y_train, y_valid, y_test = process_sample_data(raw_data)

    # Define and Compile model
    NN_model = CFAR10_model_def()

    # Train Model
    history = train_model(NN_model, X_train, y_train, X_valid, y_valid)

    # Plot Metrics
    analyze_data(history)

    # Evaluate Model on Test Set
    NN_model.evaluate(X_test, y_test)

"""
Sample Function: Function that defines the NN architecture for sample CFAR10 data-set sandbox
@Params: None
@Return: Keras sequential model
Note: This arch is effectively one VGG block
"""
def CFAR10_model_def():

    # Messing around with CFAR10
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # output model summary
    model.summary()

    # Compile model
    selected_optimizer = optimizers.Adam(lr=0.001,
                                         beta_1=0.9,
                                         beta_2=0.999)
    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer=selected_optimizer,
                     metrics=["accuracy"])

    return model


"""
Test Function: Function that tests split raw data
@Param: None
@Return: None
"""
def test_split_raw_data():

    # Generate fake data set
    print("---Testing split raw data---")
    y = np.random.randint(low = 0, high = 5, size = (10, 1))
    X = np.random.randint(low = 0, high = 5, size = (10, 2, 3, 3))
    X_train, X_val, X_test, y_train, y_val, y_test = split_raw_data(X, y, train_size=0.8, val_size=0.1, test_size=0.1)

    # Check sizes of arrays
    assert(X_train.shape == (8, 2, 3, 3))
    assert(X_val.shape == (1, 2, 3, 3))
    assert(X_test.shape == (1, 2, 3, 3))
    assert (y_train.shape == (8, 1))
    assert (y_val.shape == (1, 1))
    assert (y_test.shape == (1, 1))
    print("---Test is complete---")

"""
Sample Function: Function that defines the NN architecture for sample MNIST data-set sanbox
@Params: None
@Return: Keras sequential model
"""
def MNIST_model_def():

    # Messing around with MNIST
    model = keras.models.Sequential([
        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    # Compile model
    selected_optimizer = optimizers.Adam(lr=0.001,
                                         beta_1=0.9,
                                         beta_2=0.999)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=selected_optimizer,
                  metrics=["accuracy"])

    return model


"""
Function: Function that defines the NN architecture (based on reference paper)
@Params: None
@Return: Keras sequential model
Note: Reference architecture for CNN from https://github.com/brad-ross/crop-yield-prediction-project/tree/master/model
"""
def NN_hybrid_model_definition_():

    # Define sequential model
    model = keras.models.Sequential()

    # CNN Definition
    model.add(TimeDistributed(Conv2D(128, kernel_size=3, padding="same", activation="relu", strides=1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))
    model.add(TimeDistributed(Conv2D(128, kernel_size=3, padding="same", activation="relu", strides=1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))

    model.add(TimeDistributed(Conv2D(256, kernel_size=3, padding="same", activation="relu", strides=1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))
    model.add(TimeDistributed(Conv2D(256, kernel_size=3, padding="same", activation="relu", strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))

    model.add(TimeDistributed(Conv2D(512, kernel_size=3, padding="same", activation="relu", strides=1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))
    model.add(TimeDistributed(Conv2D(512, kernel_size=3, padding="same", activation="relu", strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))
    model.add(TimeDistributed(Conv2D(512, kernel_size=3, padding="same", activation="relu", strides=2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(rate=0.5)))

    model.add(TimeDistributed(layers.Flatten()))
    model.add(TimeDistributed(layers.Dense(2048)))  # Defaults to linear activation

    # LSTM Definition
    # TODO: Figure out the output dimension of the LSTM
    model.add(LSTM(units=20, return_sequences=True))
    model.add(Lambda(function=lambda x: K.mean(x, axis=1, keepdims=False)))

    # output model summary
    model.summary()

    # Compile model
    selected_optimizer = optimizers.Adam(lr=0.001,
                                         beta_1=0.9,
                                         beta_2=0.999)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=selected_optimizer,
                  metrics=["accuracy"])

    return model
