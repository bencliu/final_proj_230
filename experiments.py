import csv
import pandas as pd
import pickle
import rasterio
from imageProcessing import standardize_image_np, construct_tensors
from rasterio.enums import Resampling
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from imageProcessing import visualize_image, calculate_neighbors

#Functions for retrieving metadata for a specific itemID
def retrieveRow(id="20181021_162348_102e"):
    df = pd.read_csv('metadata/metacopy.csv', sep=',')
    print(df)
    row4 = df.loc[df['id'] == id] #Find row with unique id
    nprow4 = row4.to_numpy()[0] #Convert to numpy
    nprow4 = nprow4.reshape(1, 5)
    print(nprow4.shape)

#Function for writing specific row to CSV
def testWriteRow():
    fieldnames = ['id', 'anomalous_pix_perc', 'clear_percent', 'cloud_cover', 'cloud_percent', 'heavy_haze_percent',
                  'origin_x', 'origin_y', 'snow_ice_percent', 'shadow_percent']
    with open(r'metadata/metacopy.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'id':"t43", 'anomalous_pix_perc': 1, 'clear_percent':  2.0, 'cloud_cover': "test",
             'cloud_percent': 'a', 'heavy_haze_percent': 1, 'origin_x': 2.0,
             'origin_y': 3.0, 'snow_ice_percent': 1.0, 'shadow_percent': 1.0})

#Function for writing specific row to CSV
def testWriteRow2():
    fieldnames = ['id', 'anomalous_pix_perc', 'cloud_cover',
                  'origin_x', 'origin_y']
    with open(r'metadata/metadata.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'id':"t43", 'anomalous_pix_perc':1.0, 'cloud_cover': "test", 'origin_x': 2.0,
             'origin_y': 3.0})

#Resets CSV with fields
def resetCSV():
    with open(r'metadata/metacopy.csv', 'w') as f:
        #Write initial lfields
        f.write('id, anomalous_pix_perc, clear_percent, cloud_cover, cloud_percent, heavy_haze_percent, origin_x, origin_y, snow_ice_percent, shadow_percent\n')


def fetchID(id):
    df = pd.read_csv('metadata/metacopy.csv', sep=',')
    #row = df.loc[df['id'] == id]  # Find row with unique id
    #row = row.to_numpy()[0]  # Convert to numpy
    #nprow4 = row.reshape(1, 5)
    #print(nprow4)

#Change first row of metadatatest.csv file
def changeFirstRow():
    df = pd.read_csv("metadata/metacopy.csv")
    print(df.at[0, "anomalous_pix_perc"])
    print(df.at[0, "cloud_cover"])
    print(df.at[0, "origin_x"])
    print(df.at[0, "origin_y"])
    print(df)
    #df.to_csv("metacopy.csv", index=False)

def imageMosaicGen(maxH, maxW, withScale=False, scale=1.0):
    # Call image processing on downloaded item
    sat_data = rasterio.open("../ArchiveData/planet_sample1.tiff")
    image = sat_data.read(
        out_shape=(int(sat_data.height * scale), int(sat_data.width * scale)),
        resampling=Resampling.nearest
    ) \
        if withScale else sat_data.read()

    # Process and construct image
    # centered_data = standardize_image_np(image, maxH, maxW)
    b, g, r, n = image
    print("Start constructing tensors")
    image_tensor = construct_tensors(b, g, r, n)
    print("Finished constructing tensor")

    # Analyze image + store
    with open("../ArchiveData/scaled" + '.npy', 'wb') as fp:
        pickle.dump(image_tensor, fp)

def analyzeImageMosaic(npyPath):
    image = np.load(npyPath, allow_pickle=True)
    image = np.transpose(image, (1, 2, 0))

    #RGB Image
    """
    rgbTemplate = np.flip(image[:,:,:3], axis=2)
    plt.imshow(rgbTemplate.astype('uint8'))
    plt.xlabel("Latitudinal Pixel")
    plt.ylabel("Longitudinal Pixel")
    plt.show()
    """


    #NDVI, EVI, MSAVI, NDVI_NBRS
    im = image[:,:,4:5]
    nbrs = calculate_neighbors(im[:,:,0])
    displayGreen(nbrs)

def displayGreen(image):
    plt.imshow(image.astype('uint8'))
    plt.xlabel("Latitudinal Pixel")
    plt.ylabel("Longitudinal Pixel")
    plt.show()


if __name__ == "__main__":
    #imageMosaicGen(maxH=7000, maxW=7000, withScale=True, scale=0.04)
    analyzeImageMosaic("../ArchiveData/scaled" + '.npy')
    #visualize_image("../ArchiveData/planet_sample1.tiff")