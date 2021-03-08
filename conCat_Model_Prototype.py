# Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, TimeDistributed, Lambda, \
    MaxPooling2D, BatchNormalization, concatenate, Input
import pickle
import csv
import pandas as pd
from NN_Model import get_run_logdir

class ConcatDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels, n_metaFeatures, n_classes, shuffle=True):
        # Initialization
        self.dim = dim  # (H, W)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_metaFeatures = n_metaFeatures
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denote: Number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes: Called at the VERY beginning | + end of each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # Private helper method
    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples | X : (n_samples, *dim, n_channels)
        # Initialization
        image_init = np.empty((self.batch_size, *self.dim, self.n_channels))  # (numSamples, H, W, C)
        meta_init = np.empty((self.batch_size, self.n_metaFeatures, 1)) # (fields (5), 1)
        y = np.empty((self.batch_size), dtype=int)  # (numSamples) => Labels

        # read in AWS file path dictionary from pickle file
        aws_file_dict = {}
        with open('aws_file_dict_vUpdate2.p', 'rb') as fp:
            aws_file_dict = pickle.load(fp)  # dictionary of {key: id, value: aws full path}

        # read in metadata csv into dataframe
        df = pd.read_csv('metacopy.csv', sep=',')

        for i, ID in enumerate(list_IDs_temp):
            # extract image
            input_image = np.load('processed_images/concat_model/' + ID + '.npy', allow_pickle=True)

            # extract metadata
            dfrow = df.loc[df['id'] == ID]
            input_metadata = dfrow.to_numpy()[0]
            input_metadata = np.arange(5).reshape(1, 5)
            input_metadata = input_metadata[0][1:]
            input_metadata = input_metadata.reshape(1, 4)

            image_init[i,] = input_image
            meta_init[i,] = input_metadata

            # Store class
            y[i] = self.labels[ID]  # Store label

        # assemble X tuple
        X = (image_init, meta_init)

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)

    # Model architecture
class ConcatProtypeModel():
    def __init__(self):
        self.numExamples = 1000 # not actually needed in NN run
        self.width = 500
        self.height = 500
        self.numChannels = 7
        self.numMetaFeatures = 4
        self.numEpochs = 200
        self.model = None
        self.history = None
        self.validation_generator = None
        self.train_generator = None
        self.genParams = {'dim': (self.width, self.height),
                          'batch_size': 16,
                          'n_classes': 10, # not needed anymore
                          'n_channels': self.numChannels,
                          'n_metaFeatures': self.numMetaFeatures,
                          'shuffle': True}

    def generate_fake_dataset(self):
        raw_input_images = np.random.randint(low=0, high=5,
                                             size=(self.numExamples, self.height, self.width, self.numChannels))
        raw_input_metadata = np.random.randint(low=0, high=5, size=(self.numExamples, self.numMetaFeatures))
        labels = np.random.randint(low=0, high=5, size=(self.numExamples, 1))
        return raw_input_images, raw_input_metadata, labels

    def generate_fake_partition_dict(self):
        idArray = list(range(self.numExamples))
        ten_percent_split = (-1) * int(len(idArray) / 10)
        twent_percent_split = (-2) * int(len(idArray) / 10)
        test_ids = idArray[ten_percent_split:]
        val_ids = idArray[twent_percent_split:ten_percent_split]
        train_ids = idArray[:twent_percent_split]
        partition = {'train': train_ids, 'val': val_ids, 'test': test_ids}
        return partition

    def compile(self):
        # inputs
        input_images = Input(shape=(self.height, self.width, self.numChannels), name="input_images")
        input_metadata = Input(shape=(self.numMetaFeatures, 1), name="input_metadata")
        print("input_image shape per batch:" + str(input_images.get_shape()))
        print("input_metadata shape per batch:" + str(input_metadata.get_shape()))

        # image arm
        # Note: this is identical to the vanilla CNN model architecture
        im_conv2D_1_1 = Conv2D(64, 3, activation="relu", padding="same", input_shape=(self.width, self.height, 7),
                               name="image_conv2D_1_1")(input_images)
        im_conv2D_1_2 = Conv2D(64, 3, activation="relu", padding="same", name="image_conv2D_1_2")(im_conv2D_1_1)
        im_pool_1 = MaxPooling2D(2, name="image_pool_1")(im_conv2D_1_2)
        im_conv2D_2_1 = Conv2D(128, 3, activation="relu", padding="same", name="image_conv2D_2_1")(im_pool_1)
        im_conv2D_2_2 = Conv2D(128, 3, activation="relu", padding="same", name="image_conv2D_2_2")(im_conv2D_2_1)
        im_pool_2 = MaxPooling2D(2, name="image_pool_2")(im_conv2D_2_2)

        # metadata arm
        meta_dense_1 = Dense(128, activation="relu", name="meta_dense_1")(input_metadata)
        meta_drop_1 = Dropout(0.5, name="meta_drop_1")(meta_dense_1)
        meta_batch_1 = BatchNormalization(name="meta_batch_1")(meta_drop_1)
        meta_dense_2 = Dense(64, activation="relu", name="meta_dense_2")(meta_batch_1)
        meta_drop_2 = Dropout(0.5, name="meta_drop_2")(meta_dense_2)
        meta_batch_2 = BatchNormalization(name="meta_batch_2")(meta_drop_2)

        # concatenate
        flatten_images = Flatten(name="flatten_images")(im_pool_2)
        flatten_metadata = Flatten(name="flatten_metadata")(meta_batch_2)
        concat = concatenate([flatten_images, flatten_metadata], name="concat")
        cat_dense_1 = Dense(128, activation="relu", name="cat_dense_1")(concat)
        cat_drop_1 = Dropout(0.5, name="cat_drop_1")(cat_dense_1)
        cat_batch_1 = BatchNormalization(name="cat_batch_1")(cat_drop_1)
        cat_dense_2 = Dense(64, activation="relu", name="cat_dense_2")(cat_batch_1)
        cat_drop_2 = Dropout(0.5, name="cat_drop_2")(cat_dense_2)
        cat_batch_2 = BatchNormalization(name="cat_batch_2")(cat_drop_2)

        # output
        # output = Dense(10, activation = "softmax", name = "output")(cat_batch_2)
        output = Dense(1, activation="relu", name="output")(cat_batch_2)
        self.model = keras.Model(inputs=[input_images, input_metadata], outputs=[output], name="concat_model")

        # output model summary
        self.model.summary()

        # compile model
        self.model.compile(loss="mse",  # used to be categorical_crossentropy
                           optimizer=optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999),
                           metrics=["accuracy"])

    def train(self, partition, labels):
        # Generators
        self.train_generator = ConcatDataGenerator(partition['train'], labels, **self.genParams)
        self.validation_generator = ConcatDataGenerator(partition['val'], labels, **self.genParams)

        # Define model callbacks
        checkpoint_cb = callbacks.ModelCheckpoint("concat_model_results/concat_model.h5")
        early_stopping_tuning_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        run_logdir = get_run_logdir()
        tensorboard_cb = callbacks.TensorBoard(run_logdir)
        csv_cb = callbacks.CSVLogger('concat_model_results/training.log', separator=',', append=True)

        # train model
        self.history = self.model.fit(x=self.train_generator,
                                      epochs=self.numEpochs,
                                      verbose=1,
                                      validation_data=self.validation_generator,
                                      callbacks=[checkpoint_cb, early_stopping_tuning_cb, tensorboard_cb, csv_cb],
                                      shuffle=True,
                                      use_multiprocessing=True,
                                      workers=6)

    def eval(self):
        None

if __name__ == "__main__":

    """# Sample Data generation
    num_examples = 1000
    height = 50
    width = 50
    bands = 7
    metaFeatures = 5
    raw_input_images = np.random.randint(low=0, high=5, size=(num_examples, height, width, bands))
    raw_input_metadata = np.random.randint(low=0, high=5, size=(num_examples, metaFeatures))
    labels = np.random.randint(low=0, high=5, size=(num_examples, 1))

    # generate fake partition dictionary
    idArray = list(range(num_examples))
    ten_percent_split = (-1) * int(len(idArray) / 10)
    twent_percent_split = (-2) * int(len(idArray) / 10)
    test_ids = idArray[ten_percent_split:]
    val_ids = idArray[twent_percent_split:ten_percent_split]
    train_ids = idArray[:twent_percent_split]
    partition_dict = {'train': train_ids, 'val': val_ids, 'test': test_ids}"""

    # Test Model
    NN = ConcatProtypeModel()

    # Generate fake dataset for testing
    # raw_input_images, raw_input_metadata, labels = NN.generate_fake_dataset()

    # Generate fake partition
    # partition_dict = NN.generate_fake_partition_dict()

    # Define and compile NN
    NN.compile()

    # Train Model
    with open('partition_vUpdate2.p', 'rb') as fp:
        partition_file = pickle.load(fp)  # dictionary of {'train': ID list, 'val': ID list, 'test': ID list}
    with open('json_store/labels_v2/master_label_dict_vUpdate.pkl', 'rb') as fp:
        labels_file = pickle.load(fp)  # dictionary of {'id-1': label 1, ... , 'id-n', label n}
    NN.train(partition= partition_file, labels= labels_file)

    # output message
    print("------------RUN IS COMPELTE------------")
