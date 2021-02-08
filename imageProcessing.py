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
Function: Image processing
@param: Path of image TIFF file
@return: b, g, r, n np arrays used to construct tensors of human-engineered features 
"""
def simple_image_process(path):
    #Obtain image
    sat_data = rasterio.open(path)

    #Image dimensions
    width = sat_data.bounds.right - sat_data.bounds.left
    height = sat_data.bounds.top - sat_data.bounds.bottom

    # Upper left pixel
    row_min = 0
    col_min = 0

    # Lower right pixel.  Rows and columns are zero indexing.
    row_max = sat_data.height - 1
    col_max = sat_data.width - 1

    # Conversion to numpy arrays
    b, g, r, n = sat_data.read()
    return np.array([b, g, r, n])

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
    b_t = tf.convert_to_tensor(b)
    g_t = tf.convert_to_tensor(g)
    r_t = tf.convert_to_tensor(r)
    n_t = tf.convert_to_tensor(n)
    ndvi = calculate_NDVI(n_t, r_t)
    evi = calculate_EVI(b_t, r_t, n_t)
    msavi = calculate_MSAVI(r_t, n_t)
    t1 = tf.stack([b_t, g_t, r_t, n_t, ndvi, evi, msavi])
    print(t1.shape)


"""
Function: Calculate NDVI (Normalized Difference Vegetation Index)
@params: tensors for n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,1)
"""
def calculate_NDVI(n, r):
    return (n-r)/(n+r)

"""
Function: Calculate EVI (Enhanced Vegetation Index)
@params: tensors for b, n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,1)
"""
def calculate_EVI(b, r, n):
    return 2.5*((n-r)/(n+6*r-7.5*b+1))

"""
Function: Calculate MSAVI (Modified Soil Adjusted Vegetation Index)
@params: tensors for n and r, assuming shape of (m,1)
@return: NDVI as tensor with shape (m,1)
"""
def calculate_MSAVI(r, n):
    return (2*n+1-((2*n+1)^2-8*(n-r))^0.5)/2


"""
Sample Function: This is a sample test functino to create a mosaic based on .tif files from same AOI extraction. 
Note: This function is only valid if we plan to use counties as a single example - we most likely won't use this 
function but I will leave it here for reference purposes.
Reference for module: https://rasterio.readthedocs.io/en/stable/api/rasterio.merge.html
Reference for method: https://automating-gis-processes.github.io/CSC/notebooks/L5/raster-mosaic.html
@Params: File path where all the exported .tif files are located.
@Return: np.ndarry for merged rasters for the 4 bands
"""
def create_sample_mosaic(file_path = None):
    # Sample file location path
    # TODO: update path
    path = "/Users/christopheryu/Desktop/cs230\ Winter\ 2021/code/sample_images/Woodford_clippedAOI_PSOrthoTile_Explorer/files/"

    # Define file suffix
    search_criteria = "**/*_3B_AnalyticMS_clip.tif"
    # search_criteria = "**/*_BGRN_Analytic_clip.tif"
    paths = glob.glob(search_criteria, recursive=True)
    """
    Sample tif file: 20200724_162533_1040_3B_AnalyticMS_clip.tif
    Sample meta file: 20200724_162536_1040_metadata.json 
    TODO: need to figure out how to group image id's by AOI and/or date
    """

    # Create File Mosaic
    print("Starting mosaic merge...")
    start_time = time.time()
    sources = []  # list of <class 'rasterio.io.DatasetReader'>
    for file in paths:
        src = rasterio.open(file)
        sources.append(src)

    # Merge mosaic
    # [0] = numpy.ndarray of shape (# of bands, image height, image width)
    # [1] = affine.Affine (Affine transformation 3d matrix for preserving coliniearity)
    mosaic, out_transformation = merge(datasets=sources)  # outputs a tuple
    end_time = time.time()
    print("Mosaic merge completed in " + str(end_time - start_time))

    # Copy metadata for output
    out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update({"height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_transformation
                     })

    # Output image to local directory
    dirpath = "sample_images"  # temporary file path name
    out_name = "test_mosaic.tif"  # TODO: Need to come up with naming scheme for merge mosaics
    out_fp = os.path.join(dirpath, out_name)
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Extract b, g, r, n
    b = mosaic[0]
    g = mosaic[1]
    r = mosaic[2]
    n = mosaic[3]

    # return mosaic for temporary debug purposes
    return b, g, r, n

"""
Helper Function: This is a sample callable function for comparing pixels during the merge process. This pixel 
comparison function will return the pixel with the higher value.
Reference: https://github.com/mapbox/rasterio/blob/master/rasterio/merge.py
@Params: old_data (list), new_data (list), old_nodata (boolean mask), new_nodata (boolean_mask)
@Return: n/a
"""
def sample_pixel_comparison(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    mask = np.logical_and(~old_nodata, ~new_nodata)
    old_data[mask] = np.minimum(old_data[mask], new_data[mask])
    mask = np.logical_and(old_nodata, ~new_nodata)
    old_data[mask] = new_data[mask]

"""
Helper Function: This function copies a target image onto a blank reference image - the target image will be centered 
onto the reference image. The "blank reference image is a 2d numpy array of zeros with dimensions of the largest strip 
in the batch of .tif files.
@Params: target image (2d numpy array), reference image (2d zeroed numpy array)
@Return: standardized strip centered in the reference image (2d numpy array for single band)
"""
def center_images(target_image, reference_image):
    a0 = math.floor(abs((target_image.shape[0]-reference_image.shape[0]))/2)
    a1 = math.floor(abs((target_image.shape[1]-reference_image.shape[1]))/2)
    reference_image[a0:a0+target_image.shape[0],a1:a1+target_image.shape[1]] = target_image
    return reference_image

"""
Function: This code will convert all the .tif images in a file path to np arrays, and perform standardization of 
input size based on the largest strip in the batch.
*** NOTE ***: This code is a bit sloppy and can definitely use some improvements. Need to wait on 
integratedDataPipeline to come through before finalizing code and selecting data structures.
@Params: file path for where all the given .tif files are located.
@Return: original dict {image id: np.ndarray of rgbn}, standardized dict {image id: np.ndarray of rgbn}
"""
def standardize_input(file_path = None):
    # Define file search criteria
    search_criteria = "**/*_3B_AnalyticMS_clip.tif"
    paths = glob.glob(search_criteria, recursive=True)

    # open images and extract data to dictionary
    rgbn_dict = {} # TODO: instead of dictionary, use multidimensional numpy array
    sources = []
    for image_name in paths:

        # extract ID
        test_string = "20200724_162531_1040_3B_AnalyticMS_clip.tif"
        search_string = 'sample_images/Woodford_strip_clip_PSScene4Band_Explorer/files/(.*)_3B_AnalyticMS_clip.tif'
        result = re.search(search_string, image_name)
        image_ID = result.group(1)

        # Save rgbn data
        sat_data = rasterio.open(image_name)
        # source.append(sat_data) # save rasterio.io.DataSetReader for merge
        rgbn_dict[image_ID] = sat_data.read()

        # Print status message
        print("Completed conversion for " + image_ID)
        # print("Shape of rgb_dict[image_ID] is " + str(rgbn_dict[image_ID].shape))

    # identify the shape of the largest file
    max_height = 0
    max_width = 0
    for data in rgbn_dict.values():
        if data.shape[1]>max_height:
            max_height = data.shape[1]
        if data.shape[2]>max_width:
            max_width = data.shape[2]
        # print(data.shape)

    # provide padding
    max_height += 1
    max_width += 1

    # create 2d reference arrays of zeros
    reference_image = np.zeros([max_height, max_width], dtype = int)

    # standardize image against reference array
    # TODO: increase efficieny of this section
    standardized_rgbn_dict = {}
    for id, data in rgbn_dict.items(): # values numpy.ndarray (# bands, Y-dim pixel, X-dim pixel)
        bands = np.zeros((data.shape[0], max_height, max_width))
        for band_index in range(data.shape[0]):
            # standardized_band = center_images(data[band_index], reference_image)
            # rgbn_dict[id][band_index].reshape(max_height, max_width)
            # rgbn_dict[id][band_index] = standardized_band
            reference_image = np.zeros([max_height, max_width], dtype = int)
            bands[band_index] = center_images(data[band_index], reference_image)
        standardized_rgbn_dict[id] = bands
        print("Completed resize for all 4 bands of " + str(id))
    return rgbn_dict, standardized_rgbn_dict

"""
Test Function: This function will verify whether the updates images are all the same dimension
@Params: original dict {image id: np.ndarray of rgbn}, standardized dict {image id: np.ndarray of rgbn}
@Return: n/a
"""
def test_size(old, new):
    ref_key, ref_val = random.choice(list(old.items()))
    ref_height = new[ref_key].shape[1]
    ref_width = new[ref_key].shape[2]
    for key, val in old.items():
        assert(new[key][0].shape[0]==ref_height)
        assert(new[key][0].shape[1]==ref_width)
        assert(new[key][1].shape[0]==ref_height)
        assert(new[key][1].shape[1]==ref_width)
        assert(new[key][2].shape[0]==ref_height)
        assert(new[key][2].shape[1]==ref_width)
        assert(new[key][3].shape[0]==ref_height)
        assert(new[key][3].shape[1]==ref_width)
    print("numpy arrays are uniform!")


if __name__ == "__main__":
    print("Starting...")
    tiff_path = "../ArchiveData/planet_sample1.tiff"
    visualize_image(tiff_path)