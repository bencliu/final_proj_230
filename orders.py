import json
import os
import pathlib
import time

import numpy as np
import rasterio
from rasterio.plot import show
import requests
from requests.auth import HTTPBasicAuth

# global variables and setup
orders_url = 'https://api.planet.com/compute/ops/orders/v2'
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'
auth = HTTPBasicAuth(PLANET_API_KEY, '')
headers = {'content-type': 'application/json'}

#Import county extraction functions
from extractCountyData import read_county_GeoJSON

#Import download_asset functions
from downAssets import extract_images, act_download_asset, parallelize, download_file

#Import fetchData functions
from fetchData import define_county_geometry, define_filters #Filter Functions
from fetchData import search_image, get_item_asset, extract_assets, extract_items #Search Functions
from fetchData import split_into_time_series, split_by_strip, separate_by_strips

"""
Function: This function places the order based on inputted request using the order api
@Param: Request for Order API as dict (contains name, list of products, tools etc.), requests.auth.HTTPBasicAuth using PLANET_API_KEY
@Return: url of order based off of order id (string)
Directly adopted from following source:
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/tools_and_toolchains.ipynb
"""
def place_order(request, auth):
    response = requests.post(orders_url, data=json.dumps(request), auth=auth, headers=headers)
    print(response)

    if not response.ok:
        raise Exception(response.content)

    order_id = response.json()['id']
    print(order_id)
    order_url = orders_url + '/' + order_id
    return order_url

"""
Helper Function: This function will monitor and output the status of the request
@Param: url of order based off of order id (string), requests.auth.HTTPBasicAuth using PLANET_API_KEY
@Return: n/a
Directly adopted from following source:
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/tools_and_toolchains.ipynb
"""
def poll_for_success(order_url, auth, num_loops=50):
    count = 0
    while (count < num_loops):
        count += 1
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        success_states = ['success', 'partial']
        if state == 'failed':
            raise Exception(response)
        elif state in success_states:
            break
        time.sleep(10)

"""
Function: Performs product download of requests into local directory (tif, xml, udm, etc.). A folder titled "data" will 
be created and the raw data will reside in a subfolder titled by the item type.
@Param: url of order based off of order id (string), requests.auth.HTTPBasicAuth using PLANET_API_KEY, overwrite boolean flag
@Return: dictionary where keys = response names, vals = response contents
***NOTE: This function doesn't actually need to return anything, but could be a mechanism to quickly file metadata
Directly adopted from following source:
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/tools_and_toolchains.ipynb
"""
def download_order(order_url, auth, overwrite=False):
    r = requests.get(order_url, auth=auth)
    response = r.json()
    results = response['_links']['results']
    results_urls = [r['location'] for r in results] #Download URLs
    results_names = [r['name'] for r in results] #File links .tif
    results_paths = [pathlib.Path(os.path.join('data', n)) for n in results_names] #Data/pathLib
    print('{} items to download'.format(len(results_urls)))

    for url, name, path in zip(results_urls, results_names, results_paths):
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name, path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping {}'.format(path, name))

    return dict(zip(results_names, results_paths))

"""
Helper Function: Outputs sample image
@Param: File path
@Return: n/a
Directly adopted from following source:
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/tools_and_toolchains.ipynb
"""
def show_rgb(img_file):
    with rasterio.open(img_file) as src:
        b,g,r,n = src.read()
    rgb = np.stack((r,g,b), axis=0)
    show(rgb/rgb.max())






"""
Order pipeline: 
- Process all counties
- Generate tensor, crop yield pairs
- Write to google buckets
"""
def integrated_order_pipe(county_dictionary):
    #Loop through counties
    for fipCode, coordinates in county_dictionary.items():
        print("PROCESS NEW COUNTY")
        #Attain Item IDs for County
        geoFilter = define_county_geometry(coordinates)
        searchFilter = combined_filter(geoFilter)
        #id_vec = attain_itemids(searchFilter) #Deprecate

        #Gather crop statistics => Label image IDs TODO
        id_labeled_vec = process_crop_stats(searchFilter)

        #Perform ordering and downloading of images (Code in tester)

        #Multithreading Section
        #Loop through image paths => Create array, assign crop state, write object to google storage

        #Delete PSScene4Band FOlder

def thread_process_image_wrapper(imagePath, cropState):
    print("Starting thread image wrapper")


def order_and_download(itemidVector):
    print("Starting order and downloads")

"""
TODO
Helper function: Returns labeled image_ids from specified filter TODO
@Param: combined_filter
@Return: Array of image_ids with crop labeling
"""
def process_crop_stats(combined_filter):
    print("Starting to gather statistics")
    item_asset_dict = extract_assets(PLANET_API_KEY, combined_filter)
    timeSliceDict = split_into_time_series(item_asset_dict)
    finalStruct = split_by_strip(timeSliceDict)
    for time_split, strip_dict in finalStruct.item():
        numStrips = len(strip_dict.keys())

# Note: Currently commented out
def attain_itemids(combined_filter):
    item_type = "PSScene4Band"
    imageQueue = search_image(PLANET_API_KEY, combined_filter) #TODO, add combined filter
    image_ids = [feature['id'] for feature in imageQueue.json()['features']]
    print("NUMBER OF IMAGES:", len(image_ids))
    return image_ids

"""
CNN Baseline Test: 2019 Images August - November || 50% Cloud Cover
"""
def combined_filter(geojson):
    # Geo Filter
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geojson
    }

    # Date Range
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": "2019-08-01T00:00:00.000Z",
            "lte": "2019-11-01T20:00:00.000Z"
        }
    }

    # <50% cloud coverage
    cloud_cover_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lte": 0.5
        }
    }

    # combine our geo, date, cloud filters
    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter, cloud_cover_filter]
    }

    return combined_filter

if __name__ == "__main__":
    print("STARTING ORDERS PIPELINE")
    county_dictionary = read_county_GeoJSON(filename='json_store/Illinois_counties.geojson')
    integrated_order_pipe(county_dictionary)






"""
OBSELETE TESTER FUNCTIONS
"""


"""
Test Function: Sandbox for formulating requests, experimenting with tools, and performing data downloads
@Param: n/a
@Return: n/a
"""
def test_simple_download():

    # define products
    single_product = [
        {
            "item_ids": ["20151119_025740_0c74"], # ***NOTE*** Code to search for item id is already done
            "item_type": "PSScene4Band",
            "product_bundle": "analytic"
        }
    ]
    multi_product = [
        {
            "item_ids": ["20151119_025740_0c74",
                         "20151119_025739_0c74"],
            "item_type": "PSScene4Band",
            "product_bundle": "analytic"
        }
    ]

    # define clip
    clip_aoi = {
        "type": "Polygon",
        "coordinates": [[[94.81858044862747, 15.858073043526062],
                         [94.86242249608041, 15.858073043526062],
                         [94.86242249608041, 15.894323164978303],
                         [94.81858044862747, 15.894323164978303],
                         [94.81858044862747, 15.858073043526062]]]
    }
    clip = {
        "clip": {
            "aoi": clip_aoi
        }
    }

    # define delivery
    delivery = {
        "google_cloud_storage": {
            "bucket": "blah...",
            "credentials": "blah...",
            "path_prefix": "blah..."
        }
    }

    # define request
    request = {
        "name": "test download",
        "products": single_product,
        "tools": [clip]
        # "delivery": delivery
    }

    # Perform Order
    print("starting order")
    order_url = place_order(request, auth)
    print("starting polling")
    poll_for_success(order_url, auth)
    print("starting download")
    downloaded_files = download_order(order_url, auth)
    print(downloaded_files)

    for result_names, result_paths in downloaded_files.items():
        print(result_paths)
        if result_paths.endswith('_clip.tif'):
            show_rgb(result_paths)

    print("simple download test sucessful")


def test_images():
    path = 'data/48dd6063-9a8f-4e59-b866-fd4473263d1a/PSScene4Band/20151119_025740_0c74_3B_AnalyticMS_clip.tif'
    show_rgb(path)