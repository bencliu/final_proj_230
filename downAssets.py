"""
File Description: Multithreaded Asset Downloads
- Multithreaded download of mulitple assets into "images" folder
- Uses exponential backup to prevent API burnout in line with planetlabs
- Note: PlanetLabs calls this activating an item, which is basically the same as activating the assets of the item
"""

# Package imports
import os
import requests
import json
import time
import urllib.request 
import requests
from requests.auth import HTTPBasicAuth
from multiprocessing.dummy import Pool as ThreadPool
from retrying import retry
from datetime import datetime
from collections import Counter

# Import functions
from fetchData import search_image, get_item_asset, define_filters

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
            download_file(download_link)
            break


# TODO: Edit function to receive more params and pass down to download_file_nested
# Parellize activation requests
def parallelize(asset_vector):
    #Instantiate thread pool object
    parallelism = 5
    thread_pool = ThreadPool(parallelism)

    #Activate thread pool
    thread_pool.map(act_download_asset, asset_vector)

# TODO: Edit function to incorporate nested folder function
# Helper: Automatically download file given link
def download_file_nested(link="", path="./images"):
    print("DOWNLOADING FILE")
    timestr = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] #Note change this to a unique identifier
    timepath = os.path.join(path, timestr)
    urllib.request.urlretrieve(link, timepath)
    print("FINISHED DOWNLOADING FILE")


if __name__ == "__main__":
    print("starting")





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
    thread_pool.map(thread_print, nums_vec)

def thread_print(num):
    print(num)