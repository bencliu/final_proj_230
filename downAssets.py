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
from collections import Counter

# Import functions
from fetchData import search_image, get_item_asset

# Global Var
PLANET_API_KEY = "b99bfe8b97d54205bccad513987bbc02"

# Auth Setup
session = requests.Session()
session.auth = (PLANET_API_KEY, "")

"""
Helper test function:
- Extract one single image
"""
def extract_single_image(api_key):
    print("Running extract_assets")
    item_type = "PSScene4Band"
    imageQ = search_image(api_key)

    #Extract image IDs
    image_ids = [feature['id'] for feature in imageQ.json()['features']]
    print("Number of Images:", len(image_ids))

    assetResult, itemResult = get_item_asset(image_ids[0], item_type)
    return assetResult

"""
Helper function:
- Activate asset
- Download asset as image file
"""
@retry(wait_exponential_multiplier=1000,
        wait_exponential_max=10000)
def act_download_asset(assetResult):
    #Activate asset for download

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
            print("Downloading file")
            download_file(down_link)
            break

    ##TODO: Learn how to download image to folder automatically via link

# Parellize activation requests
def parallelize():
    #Instantiate thread pool object
    parallelism = 5
    thread_pool = ThreadPool(parallelism)
    asset_json_path = 'json_store/assets.txt'

    with open(asset_json_path) as file:
        assetResults = file.read().splitlines()[:4] #First 10 asset items for now

    #Activate thread pool
    thread_pool.map(act_download_asset, assetResults)

# Helper: Automatically download file given link
def download_file(link="", path="./images"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    timepath = os.path.join(path, timestr)
    urllib.request.urlretrieve(link, timepath)

# Verifies that downloading image pipeline is working properly [Single image]
def download_image_test():
    result = extract_single_image(PLANET_API_KEY)
    print("Finished extracting assets")
    link = act_download_asset(result)
    print("Downloading File")
    download_file(link)
    print("Finished downloading")

if __name__ == "__main__":
    print("starting")
    download_image_test()

