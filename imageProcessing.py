"""
File Description: Image Processing
- Load images into tensors
- Construct human-engineered features
- Construct strip vectors, time vectors
- Cloud compositing
"""

import rasterio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from rasterio import Affine
from rasterio import windows
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
import time
import re
import random
from PIL import Image

"""
Function: Visualize image
@param: path to TIFF image file
@return: np array representation of image
"""
def visualize_image(tif_path):
    image = Image.open(tif_path)
    image.show()
    imarray = np.array(image)
    print(imarray.shape)
    return imarray

"""
Function: Simple image processing
@param: Path of image TIFF file
@return: b, g, r, n np arrays used to construct tensors of human-engineered features 
"""
def simple_image_process(path):

    #Obtain image
    sat_data = rasterio.open(path)

    # Conversion to numpy arrays
    b, g, r, n = sat_data.read()

    return np.array([b, g, r, n])

"""
Function: Image processing from image to tensor
@param: Path of image TIFF file, max pixel height, max pixel width
@return: image tensor with computed vegetation indices
"""
def image_process(path, maxH, maxW):

    #Obtain image
    raw_image = rasterio.open(path)
    image = raw_image.read()

    # center data
    centered_data = standardize_image_np(image, maxH, maxW) # 2 seconds

    # Conversion to numpy arrays
    r, b, g, n = centered_data

    # Construct image tensor
    image_tensor = construct_tensors(b, g, r, n)

    return image_tensor


"""
Test function: Verifies the image process conversion from .tif to computed tensor
@Params: None
@Return: None
"""
def test_image_process():

    # sample image
    image1 = "test_image1.tif"

    # Assume inputs
    maxH = 7000
    maxW = 7000

    # run function
    start = time.time()
    std_test_image1 = image_process(image1, maxH, maxW)
    print(std_test_image1)
    end = time.time()

    # view sample output
    fig = plt.imshow(std_test_image1[0])  # print out one band
    plt.show()
    print(std_test_image1.shape)

    print("Image processing successfull:  " + str(end - start))

"""
Function: Merge Strip Vectors (After cloud processing)
@param: NP array of b, g, r, n (num_strips, 4, W, H) (4D)
@return: Merged [b, g, r, n] NP Array embedding for strip of timestep t (4, W, H) (3D)
"""
def merge_strip_vector(strip_vector):
    length = strip_vector.shape[0]
    merged = np.sum(strip_vector, axis=0) #Collapse axis 0
    return (merged / length) #Average merged arrays

"""
Function: Constructing Tensors:
@param: b, g, r, n np feeder arrays
@return: 3D tensor with human-engineered features
"""
def construct_tensors(b, g, r, n):
    b_t = tf.convert_to_tensor(b) # 0.2 seconds
    g_t = tf.convert_to_tensor(g) # 0.2 seconds
    r_t = tf.convert_to_tensor(r) # 0.2 seconds
    n_t = tf.convert_to_tensor(n) # 0.2 seconds
    ndvi = calculate_NDVI(n_t, r_t) # 2 seconds
    evi = calculate_EVI(b_t, r_t, n_t) # 2 seconds
    msavi = calculate_MSAVI(r_t, n_t) # 2 seconds
    t1 = tf.stack([b_t, g_t, r_t, n_t, ndvi, evi, msavi]) # 3 seconds
    return t1

"""
Function: Calculate NDVI (Normalized Difference Vegetation Index)
@params: tensors for n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,n)
"""
def calculate_NDVI(n, r):
    epsilon = 1e-8
    return (n-r)/(n+r+epsilon)

"""
Function: Calculate EVI (Enhanced Vegetation Index)
@params: tensors for b, n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,n)
"""
def calculate_EVI(b, r, n):
    return 2.5*((n-r)/(n+6*r-7.5*b+1))

"""
Function: Calculate MSAVI (Modified Soil Adjusted Vegetation Index)
@params: tensors for n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,n)
"""
def calculate_MSAVI(r, n):
    return 2*n+1-(tf.math.pow((2*n+1),2)-8*tf.math.pow((n-r),0.5))/2

"""
Function: This function will standardize the input based on the maximum height and width of the largest image.
@Params: image (tensor (bands, h, w)), max pixel height, max pixel width
@Return: tensor (bands, h, w)
"""
def standardize_image(image, maxH, maxW):

    # Add padding
    maxH += 1
    maxW += 1

    # Generate reference image
    reference = tf.zeros([maxH, maxW], tf.float64)

    # Compute offsets
    image_height = image.get_shape()[1]
    image_width = image.get_shape()[2]
    height_offset = math.floor(abs((image_height - maxH)) / 2)
    width_offset = math.floor(abs((image_width - maxW)) / 2)

    # Generate indices
    indices = generate_indices(height_offset, width_offset, image_height, image_width)

    # perform standardization and then stack
    std_r = tf.tensor_scatter_nd_update(reference, indices, image[0])
    std_g = tf.tensor_scatter_nd_update(reference, indices, image[1])
    std_b = tf.tensor_scatter_nd_update(reference, indices, image[2])
    std_n = tf.tensor_scatter_nd_update(reference, indices, image[3])

    # return stacked image
    return tf.stack([std_r, std_g, std_b, std_n], axis=0)

"""
Test function: Verification of the standardization algorithm for a sample image
@Params: n/a
@Return: n/a
"""
def test_standardization():

    # sample image
    image1 = "test_image1.tif"
    test_image1_raw = simple_image_process(image1)

    # Assume inputs
    maxH = 7000
    maxW = 7000
    test_image1 = tf.convert_to_tensor(test_image1_raw, np.float64)

    # run function
    start = time.time()
    std_test_image1 = standardize_image(test_image1, maxH, maxW)
    end = time.time()

    # view sample output
    fig = plt.imshow(std_test_image1[0]) # print out one band
    plt.show()
    print(std_test_image1.shape)
    print("Output has been sucessfully standardized: " + str(end - start))

def generate_indices(height_offset, width_offset, image_height, image_width):
    # start = time.time()
    indices = []
    for y in range(0, image_height):
        row = []
        for x in range(0 , image_width):
            row.append([y + height_offset, x + width_offset])
        indices.append(row)
    # end = time.time()
    # print(end - start)
    return indices

"""
Function: This function will standardize the input based on the maximum height and width of the largest image.
@Params: image (np.ndaary (bands, h, w)), max pixel height, max pixel width
@Return: tensor (bands, h, w)
Note: This is the optional version that assumes an np array input
Note: This is approximately 25x faster than the tensorflow version
"""
def standardize_image_np(image, maxH, maxW):

    # Add padding
    maxH += 1
    maxW += 1

    # Generate reference image
    reference = np.zeros((image.shape[0], maxH, maxW))

    # Compute offsets
    image_height = image.shape[1]
    image_width = image.shape[2]
    height_offset = math.floor(abs((image_height - maxH)) / 2)
    width_offset = math.floor(abs((image_width - maxW)) / 2)

    # Perform centering
    for band in range(reference.shape[0]):
        image_slice = np.zeros((maxH, maxW))
        image_slice[height_offset:height_offset + image[band].shape[0], width_offset:width_offset + image[band].shape[1]] = image[band]
        reference[band] = image_slice

    # return stacked image
    return reference

"""
Test function: Verification of the standardization algorithm for a sample image
@Params: n/a
@Return: n/a
"""
def test_standardization_np():

    # sample image
    image1 = "test_image1.tif"
    test_image1_raw = simple_image_process(image1)

    # Assume inputs
    maxH = 7000
    maxW = 7000

    # run function
    start = time.time()
    std_test_image1 = standardize_image_np(test_image1_raw, maxH, maxW)
    end = time.time()

    # view sample output
    fig = plt.imshow(std_test_image1[0]) # print out one band
    plt.show()
    print(std_test_image1.shape)

    print("Output has been sucessfully standardized: " + str(end-start))

if __name__ == "__main__":
    print("Starting...")
    """tiff_path = "../ArchiveData/planet_sample1.tiff"
    visualize_image(tiff_path)
    print(tf.version.VERSION)"""

    # test image standardization
    # test_standardization()
    # test_standardization_np()

    # test image process
    test_image_process()


