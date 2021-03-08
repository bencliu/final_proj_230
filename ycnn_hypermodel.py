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

if __name__ == "__main__":
    print("STARTING YCNN_HYPERMODEL.PY")

