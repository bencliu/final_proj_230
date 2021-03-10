"""
Class Definition:
- Defines classes for customized Resnet-50 OR VGG model architecture
- Uses dataGenerator class for custom image generation from npy files
"""

# Imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, TimeDistributed, Lambda, \
    MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras.layers import LSTM
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import kerastuner as kt
from dataGenerator import DataGenerator
import pickle5 as pickle
from NN_Model import get_run_logdir, analyze_data

class vnet_model():
    def __init__(self, width, height, imageStorePath="", numChannels=3):
        self.width = width
        self.height = height
        self.numChannels = numChannels
        self.inputShape = [self.width, self.height, self.numChannels]
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None
        self.testGenerator = None
        self.genParams = {'dim': (self.width, self.height),
                          'batch_size': 24,
                          'n_channels': self.numChannels,
                          'shuffle': True,
                          'classify': False,
                          'imageStorePath': imageStorePath}

    def compile(self, hp, useRes=True):
        print("Start Compiling...")
        # ResNet50 Definition
        resModel = ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(self.width, self.height, self.numChannels),
            pooling='max'
        )

        # VGG16 Definition
        vggModel = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(self.height, self.width, 3),
            pooling='max'
        )

        # Define model
        model = resModel if useRes else vggModel

        # Dense series definition
        flat_layer = tf.keras.layers.Flatten()(model.output)
        fc1 = keras.layers.Dense(256, activation="relu")(flat_layer)
        d1 = keras.layers.Dropout(0.5)(fc1)
        b1 = keras.layers.BatchNormalization()(d1)
        fc2 = keras.layers.Dense(128, activation="relu")(b1)
        d2 = keras.layers.Dropout(0.5)(fc2)
        b2 = keras.layers.BatchNormalization()(d2)
        fc3 = keras.layers.Dense(64, activation="relu")(b2)
        d3 = keras.layers.Dropout(0.5)(fc3)
        b3 = keras.layers.BatchNormalization()(d3)
        final = keras.layers.Dense(1, activation="relu")(b3)
        self.model = tf.keras.models.Model(inputs=resModel.input, outputs=final)

        # Hyperparameters
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        selected_optimizer = optimizers.Adam(lr=hp_learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999)
        print("Reached Pre-Compiling")
        self.model.compile(loss="mse",
                           optimizer=selected_optimizer,
                           metrics=[metrics.MeanSquaredError(),
                                    metrics.RootMeanSquaredError(),
                                    metrics.MeanAbsolutePercentageError()])
        print(self.model.summary())
        return self.model

    def train(self, labels, partition, hp=None, paramPath=None, genFake=False):
        print("Start Training...")
        if paramPath:
            self.model = keras.models.load_model(paramPath)  # load in previous model to evaluate
            return self.model

        # Define model callbacks

        checkpoint_cb = callbacks.ModelCheckpoint("resnet.h5")
        early_stopping_tuning_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        early_stopping_training_cb = callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        run_logdir = get_run_logdir()
        tensorboard_cb = callbacks.TensorBoard(run_logdir)
        csv_cb = callbacks.CSVLogger('my_logs/res_net/training.log', separator=',', append=True)


        # Tuning

        bestModel = self.model
        self.genParams['genFake'] = genFake #Testing for fake data generation
        if hp:
            hp_batch_size = hp.Choice('batch_size', values=[16, 20, 24, 28, 32])
            self.genParams['batch_size'] = hp_batch_size #Pass batch size hp to generator params
            self.train_generator = DataGenerator(partition['train'], labels, **self.genParams)
            self.validation_generator = DataGenerator(partition['val'], labels, **self.genParams)
            tuner = kt.Hyperband(self.compile,
                                 objective='val_accuracy',
                                 max_epochs=100,
                                 factor=3,
                                 directory='json_store',
                                 project_name='res_net')

            # Search for Optimal Hyperparameters: TODO, Fix this tuning procedure
            tuner.search(self.train_generator, validation_data=self.validation_generator, epochs=20,
                         callbacks=[early_stopping_tuning_cb])
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            # Train model with Optimal hyperparameters
            bestModel = tuner.hypermodel.build(best_hps)

        # Training
        history = bestModel.fit(x=self.train_generator,
                                epochs=200,
                                verbose=1,
                                validation_data=self.validation_generator,
                                callbacks=[checkpoint_cb, early_stopping_training_cb, tensorboard_cb, csv_cb],
                                shuffle=True,
                                use_multiprocessing=True,
                                workers=6)
        analyze_data(history)
        return history


    def evaluate(self, labels, partition):
        print("Start Evaluating...")
        self.testGenerator = DataGenerator(partition['test'], labels, **self.genParams)
        self.model.evaluate(x=self.testGenerator)

def generate_fake_partition_and_labels(numEx=1000):
    idArray = list(range(numEx))
    ten_percent_split = (-1) * int(len(idArray) / 10)
    twent_percent_split = (-2) * int(len(idArray) / 10)
    test_ids = idArray[ten_percent_split:]
    val_ids = idArray[twent_percent_split:ten_percent_split]
    train_ids = idArray[:twent_percent_split]
    partition = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    labels = np.random.randint(low=0, high=5, size=(numEx, 1))
    return labels, partition

def test_vnet_fake():
    model = vnet_model(width=500, height=500)
    hp = kt.HyperParameters()
    model.compile(hp=hp)
    labels, partition =generate_fake_partition_and_labels()
    model.train(labels, partition, hp=hp, genFake=True)

if __name__ == "__main__":
    print("STARTING...")
    #test_vnet_fake