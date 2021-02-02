# Imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, TimeDistributed, Lambda
from keras.layers import LSTM
from keras import callbacks
import tensorflow.keras.backend as K
import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

def acquire_raw_data(raw_data):
    # Note: Placeholder function for acquiring satellite data
    # -------------------------------------------------------
    # Inputs: TBD Raw Data (using MNIST as bogus example for Jan 31)
    # Return: ((x_train, y_train), (x_test, y_test))
    # x is unit8 array of RBG with shape (num_samples, 32, 32, 3)
    # y is unit8 category labels (0-9) with shape (num_samples, 1)
    return tf.keras.datasets.mnist.load_data()

def process_raw_data(acquired_data):
    # Note: Placeholder function for acquiring satellite data
    # -------------------------------------------------------
    # Inputs: Tuple of NP arrays (x_train, y_train), (x_test, y_test)
    # Outputs: X_train, X_valid, X_test, y_train, y_valid, y_test
    
    # Isolate tuples 
    full_training = acquired_data[0]
    full_test = acquired_data[1]
    X_training = full_training[0]
    y_training = full_training[1]
    X_test = full_test[0]
    y_test = full_test[1]
    
    # Perform Pre-processing (cloud removal, normalization, etc.)
    X_training = X_training/255.
    X_test = X_test/255.
    # TODO: Add in other pre-processing techniques
    
    # Split training data into train/validation
    X_train, X_valid, y_train, y_valid = split_data(X_training, y_training)
    
    # Reshape arrays
    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    y_valid = y_valid[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    
    # Verify np array dimensions
    """
    train_percent = 0.9 # delete later
    print(X_train.shape)
    assert(X_train.shape == (54000, 28, 28, 1))
    assert(X_valid.shape == (6000, 28, 28, 1))
    assert(X_test.shape == (10000, 28, 28, 1))
    assert(y_train.shape == (54000, 1))
    assert(y_valid.shape == (6000, 1))
    assert(y_test.shape == (10000, 1))
    """
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
    
def split_data(X_train_full, y_train_full):
    # Note: This function splits training set into train/val
    # Constants:
    train_percent = 0.9 # set percent training data
    # -------------------------------------------------------
    # Inputs: Train X, Train y
    # Outputs: Train X, Val X, Train Y, Val y
    
    # Compute number of samples to split up
    num_samples = X_train_full.shape[0]
    num_val_samples = math.floor(1-train_percent*num_samples)
    
    # perform split
    X_train, X_valid = X_train_full[:-num_val_samples], X_train_full[-num_val_samples:]
    y_train, y_valid = y_train_full[:-num_val_samples], y_train_full[-num_val_samples:]
    
    return X_train, X_valid, y_train, y_valid

def NN_architecture_definition():
    # Note: Reference architecture for CNN from https://github.com/brad-ross/crop-yield-prediction-project/tree/master/model
    # -------------------------------------------------------
    
    # Inputs: None
    # Outputs: Fully defined NN sequential model
    
    """
    # Define sequential model
    model = keras.models.Sequential()
        
    # CNN Definition
    # TODO: Figure out the appropriate CNN architecture, especially the desired output
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
    model.add(TimeDistributed(layers.Dense(2048))) # Defaults to linear activation
        
    # LSTM Definition
    # TODO: Figure out the output dimension of the LSTM
    model.add(LSTM(units=20, return_sequences=True))
    model.add(Lambda(function=lambda x: K.mean(x, axis=1, keepdims=False)))
    """
    
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

    # output model summary
    # model.summary()
        
    return model

def compile_model():
    # Note: This function defines and compiles NN architecture
    # Inputs: n/a
    # Outputs: compiled NN model
    # -------------------------------------------------------
    NN_model = NN_architecture_definition()
    selected_optimizer = optimizers.Adam(lr=0.001,
                                         beta_1=0.9,
                                         beta_2=0.999)
    NN_model.compile(loss="sparse_categorical_crossentropy",
                     optimizer=selected_optimizer,
                     metrics=["accuracy"])
    return NN_model

def train_model(NN_model, X_train, y_train, X_valid, y_valid):
    # Note: This function trains NN
    # Inputs: compiled NN, X train, y train, X val, y val
    # Outputs: History of trained model
    # -------------------------------------------------------
    # TODO: add more callbacks
    checkpoint_cb = callbacks.ModelCheckpoint("TBD_name_model.h5")
    early_stopping_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    run_logdir = get_run_logdir()
    tensorboard_cb = callbacks.TensorBoard(run_logdir)
    history = NN_model.fit(X_train,
                           y_train,
                           epochs=5,
                           validation_data=(X_valid, y_valid),
                           callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    return history

def get_run_logdir():
    # Note: This function creates path for NN run
    # Inputs: n/a
    # Outputs: path for NN run
    # -------------------------------------------------------
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    root_logdir = os.path.join(os.curdir, "my_logs")
    return os.path.join(root_logdir, run_id)

def analyze_data(history):
    # Note: This is a placeholder function for any sort of data analysis
    # Inputs: History object after training model
    # Outputs: none
    # -------------------------------------------------------
    # TODO: Add other analyses

    # Added bogus plot for validation lost
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    
    # run logistics
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # acquire raw satellite imagery from API
    raw_data = acquire_raw_data(None)
    
    # apply pre-processing (e.g. cloud removal, rbg normatlization, etc.)
    X_train, X_valid, X_test, y_train, y_valid, y_test = process_raw_data(raw_data)
    
    # Define and Compile model
    NN_model = compile_model()
    
    # Train Model
    history = train_model(NN_model, X_train, y_train, X_valid, y_valid)

    # Plot Metrics
    analyze_data(history)
    
    # Evaluate Model on Test Set
    # NN_model = keras.models.load_model("TBD_name_model.h5") # load in previous model to evaluate
    NN_model.evaluate(X_test, y_test)
    


