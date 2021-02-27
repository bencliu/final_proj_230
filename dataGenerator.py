import numpy as np
from tensorflow import keras
import io
import boto3
import pickle

# Import functions
from imageProcessing import image_process

# Define global variables
AWS_SERVER_PUBLIC_KEY = "" #TODO: Add after pull
AWS_SERVER_SECRET_KEY = "" #TODO: Add after pull

"""
Class: The DataGenerator class allows for data to be generated in parallel by CPU and fed into GPU real time.
Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

TODO Notes:
1. Data Generator Class is 100%
2. Partition and label sets returned from the cloud_util method
3. Line 60 needs to be changed to load npy file from AWS
"""

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(951,951), n_channels=7,
                 n_classes=10, shuffle=True):
        #Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #Denote: Number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        #Updates indexes: Called at the VERY beginning | + end of each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    #Private helper method
    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples | X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) # (numSamples, H, W, C)
        y = np.empty((self.batch_size), dtype=int) # (numSamples) => Labels

        # read in AWS file path dictionary from pickle file
        aws_file_dict = {}
        with open('aws_file_dict.p', 'rb') as fp:
            aws_file_dict = pickle.load(fp)  # dictionary of {key: id, value: aws full path}

        # Generate data
        session = boto3.Session(
            aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
            aws_secret_access_key=AWS_SERVER_SECRET_KEY,
        )
        for i, ID in enumerate(list_IDs_temp):
            print("GENERATING NEW IMAGE <GATA_GENERATION> Iteration", i)
            # Store sample

            # process image
            maxH = 950
            maxW = 950
            X_data = image_process(session, 's3://cs230data/' + aws_file_dict[ID], maxW=maxW, maxH=maxH)  # shape (C, H, W)
            X_data = np.transpose(X_data, (1, 2, 0))  # shape (H, W, C)
            assert X_data.shape == (maxW+1, maxH+1, 7)
            X[i,] = X_data

            # Store class
            y[i] = self.labels[ID] - 1 #Store label

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)