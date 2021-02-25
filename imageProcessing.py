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
from scipy import sparse
import boto3

# global variables and setup
orders_url = 'https://api.planet.com/compute/ops/orders/v2'
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'
AWS_SERVER_PUBLIC_KEY = "" #TODO: Add after pull
AWS_SERVER_SECRET_KEY = "" #TODO: Add after pull

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
**Integrated function**
Function: Image processing from image to tensor
@param: Path of image TIFF file, max pixel height, max pixel width
@return: image np.ndarray with computed vegetation indices with shape (# channels, H, W)
TODO: Change max height and max width parameters
"""
def image_process(session, path, maxH=6000, maxW=8300):

    with rasterio.Env(aws_secret_access_key=AWS_SERVER_SECRET_KEY,
                      aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
                      aws_session=session):
        data = rasterio.open(path)
        image = data.read()

        # center data
        centered_data = standardize_image_np(image, maxH, maxW) # 2 seconds

        # split numpy arrays
        b, g, r, n = centered_data

        # Construct image tensor
        image_tensor = construct_tensors(b, g, r, n)

    return image_tensor

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

    """ # convert to sparse matrices
    # Note using scipy sparse matrix only cuts down on time by 20% or so
    b = sparse.csr_matrix(b)
    g = sparse.csr_matrix(g)
    r = sparse.csr_matrix(r)
    n = sparse.csr_matrix(n)"""

    np.seterr(divide='ignore', invalid='ignore')
    ndvi = calculate_NDVI(n, r)
    evi = calculate_EVI(b, r, n)
    msavi = calculate_MSAVI(r, n)
    t1 = np.stack((b, g, r, n, ndvi, evi, msavi)) # 3 seconds
    # t1 = tf.convert_to_tensor(stacked)

    """b_t = tf.convert_to_tensor(b) # 0.2 seconds
    g_t = tf.convert_to_tensor(g) # 0.2 seconds
    r_t = tf.convert_to_tensor(r) # 0.2 seconds
    n_t = tf.convert_to_tensor(n) # 0.2 seconds
    ndvi = calculate_NDVI(n_t, r_t) # 2 seconds
    evi = calculate_EVI(b_t, r_t, n_t) # 2 seconds
    msavi = calculate_MSAVI(r_t, n_t) # 2 seconds
    t1 = tf.stack([b_t, g_t, r_t, n_t, ndvi, evi, msavi]) # 3 seconds"""

    return t1

"""
Function: Calculate NDVI (Normalized Difference Vegetation Index)
@params: tensors for n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,n)
"""
def calculate_NDVI(n, r):
    epsilon = 1e-8
    # tensor_result = (n-r)/(n+r+epsilon)
    np_result = np.divide(n-r, n+r+epsilon)
    return np_result

"""
Function: Calculate EVI (Enhanced Vegetation Index)
@params: tensors for b, n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,n)
"""
def calculate_EVI(b, r, n):
    # tensor_result = 2.5*((n-r)/(n+6*r-7.5*b+1))
    np_result = 2.5*(np.divide(n-r, n+6*r-7.5*b+1))
    return np_result

"""
Function: Calculate MSAVI (Modified Soil Adjusted Vegetation Index)
@params: tensors for n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,n)
"""
def calculate_MSAVI(r, n):
    # tensor_result = (2*n+1-tf.math.pow((2*n+1),2)-8*tf.math.pow((n-r),0.5))/2
    np_result = (2*n+1-np.power((2*n+1),2)-8*np.power((n-r),0.5))/2
    return np_result

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

    # Compute offsets
    image_height = image.shape[1]
    image_width = image.shape[2]
    height_offset = math.floor(abs((image_height - maxH)) / 2)
    width_offset = math.floor(abs((image_width - maxW)) / 2)

    # Generate reference image
    reference = np.zeros((image.shape[0], maxH, maxW))

    # Perform centering
    for band in range(reference.shape[0]):
        image_slice = np.zeros((maxH, maxW))
        image_slice[height_offset:height_offset + image[band].shape[0], width_offset:width_offset + image[band].shape[1]] = image[band]
        # image_slice = np.pad(image[band], (height_offset, width_offset), 'constant', constant_values = 0)
        # print(image_slice.shape)
        reference[band] = image_slice

    # return stacked image
    return reference


"""
Test function: Verifies the image process conversion from .tif to computed tensor using s3
@Params: None
@Return: None
"""
def test_cloud_image_process():

    # Define session for aws
    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )

    # sample image
    image1 = 's3://cs230data/test_image1.tif'

    # Assume inputs
    maxH = 5000
    maxW = 5000

    # run function
    print("Starting image standardization")
    start = time.time()
    std_test_image1 = image_process(session, image1, maxH, maxW)
    end = time.time()

    # view sample output
    fig = plt.imshow(std_test_image1[0])  # print out one band
    plt.show()
    print(std_test_image1.shape)
    print(type(std_test_image1))

    print("Image processing successfull:  " + str(end - start))


if __name__ == "__main__":
    print("Starting...")
    """tiff_path = "../ArchiveData/planet_sample1.tiff"
    visualize_image(tiff_path)
    print(tf.version.VERSION)"""

    # test image standardization
    # test_standardization()
    # test_standardization_np()
    # test_image_process()
    test_cloud_image_process()



"""
Obselete Test Functions
"""

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