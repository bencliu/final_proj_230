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
Engineered features: TODO 
"""
def construct_tensors(b, g, r, n):
    bt = tf.convert_to_tensor(b)
    gt = tf.convert_to_tensor(g)
    rt = tf.convert_to_tensor(r)
    nt = tf.convert_to_tensor(n)
    t1 = tf.stack([bt, gt, rt, nt])
    print(t1.shape)

if __name__ == "__main__":
    print("Starting...")
    tiff_path = "../ArchiveData/planet_sample1.tiff"
    visualize_image(tiff_path)