import numpy as np
import keras

"""
Class: The DataGenerator class allows for data to be generated in parallel by CPU and fed into GPU real time.
Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

TODO Notes:
1. Data Generator Class is 100%
2. Partition and label sets returned from the cloud_util method
3. Line 60 needs to be changed to load npy file from AWS
"""

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(8000,8000), n_channels=7,
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

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy') #Load data from npy file (Already instantiated), sample_number dimensions specified | H, W, C follow implicitely

            # Store class
            y[i] = self.labels[ID] #Store label

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)