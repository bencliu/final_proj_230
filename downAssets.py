"""
File Description: Multithreaded Asset Downloads
- Multithreaded download of mulitple assets into "images" folder
- Uses exponential backup to prevent API burnout in line with planetlabs
- Note: PlanetLabs calls this activating an item, which is basically the same as activating the assets of the item
"""

# Package imports
import os
import json
import time
import urllib.request 
import requests
from requests.auth import HTTPBasicAuth
from multiprocessing.dummy import Pool as ThreadPool
from retrying import retry
from datetime import datetime
import rasterio
import tensorflow as tf
import numpy as np
from collections import Counter

# Import functions
from fetchData import search_image, get_item_asset, define_filters
from imageProcessing import construct_tensors

# Global Var
PLANET_API_KEY = "b99bfe8b97d54205bccad513987bbc02"

# Auth Setup
session = requests.Session()
session.auth = (PLANET_API_KEY, "")

"""
Helper test function:
- Extract one or more images
@Param: API Key and # Images to Extract
@Return: Vector of assetResults corresponding to images
"""
def extract_images(api_key, num_images, combined_filter):
    print("Running extract_assets")
    item_type = "PSScene4Band"
    imageQ = search_image(api_key, combined_filter)

    #Extract image IDs
    image_ids = [feature['id'] for feature in imageQ.json()['features']]
    print("Number of Images:", len(image_ids))

    assetVector = []
    for index in range(num_images):
        assetResult, itemResult = get_item_asset(image_ids[index], item_type)
        assetVector.append(assetResult)
    return assetVector

"""
Helper function:
- Activate asset
- Download asset as image file
"""
@retry(wait_exponential_multiplier=1000,
        wait_exponential_max=10000)
def act_download_asset(assetResult):
    #Activate asset for download
    print("Start activating and downloading asset")

    # Parse out useful links
    links = assetResult.json()[u"analytic"]["_links"]
    self_link = links["_self"] #Check activation status
    activation_link = links["activate"] #Sent to activate asset

    # Request activation of the 'analytic' asset:
    activate_result = \
        session.get(
            activation_link,
            auth=HTTPBasicAuth(PLANET_API_KEY, '')
        )

    #Trigger retry
    if activate_result.status_code == 429:
        print("Exception reached")
        raise Exception("rate limit error")

    # Wait until activation complete, then print download link
    while True:
        activation_status_result = \
            session.get(
                self_link,
                auth=HTTPBasicAuth(PLANET_API_KEY, '')
            )
        if activation_status_result.json()["status"] == 'active':
            download_link = activation_status_result.json()["location"]
            return download_link
            break
"""
Wrapper function for parallelization:
@Param: Download_link for image
@Return: Image tensor
"""
def download_process_wrapper(asset):
    link = act_download_asset(asset)
    image_tensor = download_process_image(link)
    return image_tensor

# Parellize activation requests
def parallelize(asset_vector):
    print("START PARALLELIZING")
    #Instantiate thread pool object
    parallelism = 5
    thread_pool = ThreadPool(parallelism)

    #Activate thread pool
    imageTensorArray = thread_pool.map(download_process_wrapper, asset_vector)
    print("COMPLETE PARALLELIZING")
    print(imageTensorArray)
    return imageTensorArray

# Helper: Automatically download file given link
def download_process_image(link="", path="./images"):
    #Download File
    print("DOWNLOADING FILE")
    timestr = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] #Note change this to a unique identifier
    timepath = os.path.join(path, timestr)
    urllib.request.urlretrieve(link, timepath)
    print("FINISHED DOWNLOADING FILE")

    #Create image tensor
    print("START PROCESSING IMAGE")
    sat_data = rasterio.open(timepath)
    b, g, r, n = sat_data.read()
    imageTensor = construct_tensors(b, g, r, n) #TODO, standardize images here [Function is wrong]
    print("COMPLETE IMAGE PROCESSING")


    #Delete file
    print("DELETING FILE, FINISHED PROCESSING")
    os.remove(timepath)
    return imageTensor








"""
Test Function Queue:
- Note these functions are obselete
- The results of these functions have already been integrated into helper functions
"""

# Helper: Automatically download file given link
def download_file(link="", path="./images"):
    print("DOWNLOADING FILE")
    timestr = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] #Note change this to a unique identifier
    timepath = os.path.join(path, timestr)
    urllib.request.urlretrieve(link, timepath)
    print("FINISHED DOWNLOADING FILE")

# Verifies that downloading image pipeline is working properly [Single image]
def download_image_test():
    combined_filter = define_filters()
    result = extract_images(PLANET_API_KEY, 1, combined_filter)[0] #Single image vector
    print("Finished extracting assets")
    link = act_download_asset(result)

# Verifies parallelizing requests, threadpool is working properly
def thread_download_test():
    combined_filter = define_filters()
    assetResultsVector = extract_images(PLANET_API_KEY, 8, combined_filter)
    print("Finished obtaining asset_results")
    parallelize(assetResultsVector)


# Tests outputted numpy array image conversion shapes
def test_image_dimensions():
    #thread_download_test()
    directory = "images"
    print("Starting to convert images to numpy arrays")
    for file in os.listdir(directory):  # Loops through images
        filename = os.fsdecode(file)
        path = os.path.join(directory, filename)
        sat_data = rasterio.open(path)
        b, g, r, n = sat_data.read()
        combined = np.array([b, g, r, n])
        print(combined.shape)

# Tests Parallel Processing: Simultaneous Image Downloading, Processing, Deletions
def test_multithreaded_processing():
    combined_filter = define_filters()
    assetResultsVector = extract_images(PLANET_API_KEY, 8, combined_filter)
    results = parallelize(assetResultsVector)
    for tensor in results:
        print(tensor.shape)

"""
Thread Pool Experiments:
- Tester function for parallelizing
- Tester print function
- Tester integration function
"""

def test_parallel():
    parallelism = 5
    nums_vec = [1, 2, 3]
    thread_pool = ThreadPool(parallelism)

    # Activate thread pool
    results = thread_pool.map(thread_print, nums_vec)
    print(results)

def thread_print(num):
    return num




if __name__ == "__main__":
    print("Starting downAssets...")
    test_multithreaded_processing()
