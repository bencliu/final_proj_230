import json
import os
import pathlib
import time

import numpy as np
import rasterio
from rasterio.plot import show
import requests
import pickle
from requests.auth import HTTPBasicAuth
import datetime
import os.path
import time
import pprint

# global variables and setup
orders_url = 'https://api.planet.com/compute/ops/orders/v2'
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'
AWS_SERVER_PUBLIC_KEY = "" #TODO: Add after pull
AWS_SERVER_SECRET_KEY = "" #TODO: Add after pull
auth = HTTPBasicAuth(PLANET_API_KEY, '')
headers = {'content-type': 'application/json'}

#Import county extraction functions
from extractCountyData import read_county_GeoJSON, read_county_truth

#Import download_asset functions
from downAssets import extract_images, act_download_asset, parallelize, download_file

#Import fetchData functions
from fetchData import define_county_geometry, define_filters #Filter Functions
from fetchData import search_image, get_item_asset, extract_assets, extract_items #Search Functions
from fetchData import split_into_time_series, split_by_strip, separate_by_strips

"""
High_Level Notes:
- ResultName is '41ac7935-979b-4014-b15b-11d6db775164/PSScene4Band/20151119_025740_0c74_3B_AnalyticMS_metadata_clip.xml'
- Text after item_type is item_id (Before "analytic" in above example"
"""

"""
SECTION: Core functionality for orders: 1) Request sending, 2) Queueing, 3) Downloading
"""

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
    print("RESULT NAMES:", results_names)
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
SECTION: Core functionality for 1) Integrated order pipeline; 2) Order Filtering
"""

"""
Order pipeline: 
- Process all counties
- Write images to AWS buckets 
"""
def integrated_order_pipe(county_dictionary):
    #Loop through counties
    for fipCode, coordinates in county_dictionary.items():
        print("PROCESS NEW COUNTY:" + str(fipCode))
        #Attain Item IDs for County
        geoFilter = define_county_geometry(coordinates)
        searchFilter = combined_filter(geoFilter)
        id_vec = attain_itemids(searchFilter)

        #Perform ordering and downloading of images (Code in tester) (Write to AWS)
        order_and_download(id_vec, fipCode, coordinates)
        print("COMPLETED S3 DOWNLOAD FOR {}".format(fipCode))

def order_and_download(itemidVector, fipCode, coordinates):
    print("Starting order and downloads")
    # define products
    single_product = [
        {
            "item_ids": itemidVector,
            "item_type": "PSScene4Band",
            "product_bundle": "analytic"
        }
    ]

    # define clip
    clip_aoi = {
        "type": "Polygon",
        "coordinates": coordinates
    }

    clip = {
        "clip": {
            "aoi": clip_aoi
        }
    }

    path_prefix = "county" + str(fipCode)
    # define delivery
    delivery = {
        "amazon_s3": {
            "bucket": "cs230data",
            "aws_region": "us-west-1",
            "aws_access_key_id": AWS_SERVER_PUBLIC_KEY,
            "aws_secret_access_key": AWS_SERVER_SECRET_KEY,
            "path_prefix": path_prefix
        }
    }

    # define request
    request = {
        "name": "County Download Test1",
        "products": single_product,
        "tools": [clip],
        "delivery": delivery
    }

    # Perform Order
    print("starting order")
    order_url = place_order(request, auth)
    print("starting polling")
    poll_for_success(order_url, auth)
    print("Completed Order and Download to AWS")


"""
SECTION: Core functionality for LABELING: 
1) Obtaining crop yields in master dictionary
2) Processing crop statistics for each county
3) Attaining yield for designated time splits
"""

"""
Integrated Function:
- Obtains master dictionary of itemIDs to crop yields
- Calls process_crop_stats helper for each county 
@ Params: 
    - county_dictionary of dict[County FIP: List[List[Coordinates]]]
    - county_truth of dict[County FIP: dict[year, yield]]
@ Return: Pickle files in local directory
"""
def obtain_crop_labels(county_dictionary, county_truth):
    master_yield_id_dictionary = {}

    # read back in saved crop list
    completed_fips = []
    if (os.path.isfile('json_store/labels/completed_fips.pkl')):
        with open('json_store/labels/completed_fips.pkl', 'rb') as fp:
            completed_fips = pickle.load(fp)

    #Loop through counties
    for fipCode, coordinates in county_dictionary.items():

        # Assuming restart after failed run, check to see if fip has already been processed
        if (fipCode in completed_fips):
            continue # skip fip if already processed

        # process county
        print("PROCESS NEW COUNTY: " + str(fipCode))
        start = time.time()
        #Attain Item IDs for County
        geoFilter = define_county_geometry(coordinates)
        searchFilter = combined_filter(geoFilter)
        county_dict = process_crop_stats(searchFilter, fipCode, county_truth)

        # write save county dict (key: id, value: yields)
        ids = list(county_dict.keys())
        with open('json_store/labels/' + str(fipCode) + '.pkl', 'wb') as fp:
            pickle.dump(county_dict, fp)
        # master_yield_id_dictionary = {**master_yield_id_dictionary, **county_dict} # Need to be compatabile with 3.8

        # store completed FIPS
        completed_fips.append(fipCode)
        with open('json_store/labels/completed_fips.pkl', 'wb') as fp:
            pickle.dump(completed_fips, fp)

        # output timed data
        end = time.time()
        print("County " + str(fipCode) + " complete in " + str(end-start))
        print(datetime.datetime.now().time())

"""
Function: This method reads the generated pickle files into the label dictionary
Note: Only run this funcntion once all the local pickle files have been generated
@Params: None
@Return: Label Dictionary of dict{key: id, value: yield per image}
"""
def read_pickle_crop_labels():

    # read in list of FIPS
    completed_fips = []
    with open('json_store/labels/completed_fips.pkl', 'rb') as fp:
        completed_fips = pickle.load(fp)

    # test for overlapping counties
    # completed_fips = [77, 87]
    # completed_fips = [77, 181]

    # read in all county label dictionary files
    master_label_dict = {}
    for fipCode in completed_fips:
        county_label_dict = {}
        with open('json_store/labels/' + str(fipCode) + '.pkl', 'rb') as fp:
            county_label_dict = pickle.load(fp)
            master_label_dict = {**master_label_dict, **county_label_dict}

    # write master pickle file to local directory
    with open('json_store/labels/master_label_dict.pkl', 'wb') as fp:
        pickle.dump(master_label_dict, fp)

    return master_label_dict

"""
Helper function: Returns dictionary of image_ids corresponding to crop yield labels 
@Param: 
    - combined_filter dict
    - fip_code int
    - county_truth dict[County FIP: dict[year, yield]]
@Return: Array of image_ids with crop labeling
"""
def process_crop_stats(combined_filter, fip_code, county_truth):
    print("STARTING TO GATHER CROP STATISTICS")
    item_asset_dict = extract_items(PLANET_API_KEY, combined_filter)
    timeSliceDict = split_into_time_series(item_asset_dict) #Split into months

    #Dictionary Time Split : {Dictionary of StripID : Vector of Activated Items}
    finalStruct = split_by_strip(timeSliceDict, singleItem=True)
    image_to_yield_dict = {}

    #Loop through time splits
    for time_split, strip_dict in finalStruct.items():
        #Gathered from groundtruth
        cropYieldInTimeSplit = yield_for_time_split(time_split, fip_code, county_truth)
        numStrips = len(strip_dict.keys())

        #Yield for each strip
        yieldPerStrip = cropYieldInTimeSplit / numStrips
        # print("YIELD PER STRIP:", yieldPerStrip)

        #Loop through each strip
        for stripid, itemVec in strip_dict.items():
            #Obtain yield per each image **Naively depends on num images**
            numImages = len(itemVec)
            yieldPerImage = yieldPerStrip / numImages
            # print("YIELD PER image:", yieldPerImage)
            #Assign crop yields to each image: Update dictionary with itemID, yield
            for image in itemVec:
                #TODONOW, Currently Debugging
                prop = image.json()
                # print("PROP:", prop)
                # print("ACQUIRED:", image.json()['properties']['acquired'])
                itemid = image.json()['id'] #Can add other metadata from the itemResult
                # print(itemid)
                image_to_yield_dict[itemid] = yieldPerImage

    return image_to_yield_dict

"""
Return crop yield for given time split
@Param: 
    - Date time object
    - fip code int 
    - county_truth dict[County FIP: dict[year, yield]]
@Return: Crop yield (Float) for given time split
"""
def yield_for_time_split(dateTime, fipCode, county_truth):
    year = dateTime.year
    yearYield = 0

    #yearArray = county_truth[fipCode].keys() #Year list only
    yearArray = list(county_truth[fipCode].keys())  # Year list only

    if year not in yearArray: #Not in year array, take averages of left and right neighbors
        leftYear = int(year) - 1
        rightYear = int(year) + 1

        if year > max(yearArray): #Already at the right border
            rightYear = yearArray[0]
            leftYear = yearArray[1]
        elif year < min(yearArray):
            leftYear = yearArray[-1]
            rightYear = yearArray[-2]
        else:
            year = leftYear, rightYear
            while leftYear not in yearArray:
                leftYear -= 1
            while rightYear not in yearArray:
                rightYear += 1
        yearYield = (county_truth[fipCode][leftYear] + county_truth[fipCode][rightYear]) / 2 #Average of left and right neighbors
    else:
        yearYield = county_truth[fipCode][year]

    return yearYield / 12

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

    """
    print("STARTING ORDERS PIPELINE")
    county_dictionary = read_county_GeoJSON(filename='json_store/Illinois_counties_visvalingam_weighted_20.geojson')
    integrated_order_pipe(county_dictionary)
    # test function
    #test_process_crop_statistics(dict)
    """

    label_dict = read_pickle_crop_labels()
    print(len(label_dict))

    """print("Starting to collect labels")
    county_dict = read_county_GeoJSON(filename='json_store/Illinois_counties.geojson')
    county_truth = read_county_truth(filename='json_store/Illinois_Soybeans_Truth_Data.csv')
    obtain_crop_labels(county_dict, county_truth)
    print("Finished collecting labels")"""

    """path = "json_store/labels_all"
    outfile = open(path, 'wb')
    pickle.dump(cropLabels, outfile)
    outfile.close()"""










"""
OBSELETE TESTER FUNCTIONS
"""

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
        if str(result_paths).endswith('MS_clip.tif'):
            show_rgb(result_paths)

    print("simple download test sucessful")
    
def test_images():
    path = 'data/48dd6063-9a8f-4e59-b866-fd4473263d1a/PSScene4Band/20151119_025740_0c74_3B_AnalyticMS_clip.tif'
    show_rgb(path)

def test_process_crop_statistics(image_to_yield_dict):

    # Test Case
    fip_code = 1
    dateTime_sample = datetime.datetime(2019, 10, 15)

    # Create truth dictionaray and confirm yield
    county_truth = read_county_truth("json_store/Illinois_Soybeans_Truth_Data.csv")
    yearYield = yield_for_time_split(dateTime_sample, fip_code, county_truth)
    assert (yearYield * 12 == 49.8)

    # Verify yield per strip calculations
    assert(round(image_to_yield_dict["20191012_150709_1049"] * 3 * 9, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191012_150714_1049"] * 3 * 9, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191012_162948_1035"] * 3 * 4, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191012_162955_1035"] * 3 * 4, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191013_164837_82_1057"] * 2 * 5, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191013_164845_96_1057"] * 2 * 5, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191014_150319_1054"] * 4 * 3, 2) == round(yearYield, 2))
    assert(round(image_to_yield_dict["20191014_150321_1054"] * 4 * 3, 2) == round(yearYield, 2))

"""
Test function: Verifies the yield_for_time_split function
"""
def test_yield_for_time_split():

    print("starting test for yield_for_time_split function")

    # Test parameters
    dateTime_sample = datetime.datetime(2019, 5, 17)
    FIPS_sample = 1 # Adams County

    # Create truth dictionaray
    county_truth = read_county_truth("json_store/Illinois_Soybeans_Truth_Data.csv")
    yearYield = yield_for_time_split(dateTime_sample, FIPS_sample, county_truth)
    assert(yearYield*12 == 49.8)

    print("Simple test for yield_for_time_split complete")

"""
Debug Function: debugging yield_for_time_split
"""
def debug_yield_for_time_split():
    # Test Case
    fip_code = 3
    dateTime_sample = datetime.datetime(2013, 10, 15)
    county_truth = read_county_truth("json_store/Illinois_Soybeans_Truth_Data.csv")

    print(yield_for_time_split(dateTime_sample, fip_code, county_truth))

    def test_process_crop_statistics(image_to_yield_dict):
        # Test Case
        fip_code = 1
        dateTime_sample = datetime.datetime(2019, 10, 15)

        # Create truth dictionaray and confirm yield
        county_truth = read_county_truth("json_store/Illinois_Soybeans_Truth_Data.csv")
        yearYield = yield_for_time_split(dateTime_sample, fip_code, county_truth)
        assert (yearYield * 12 == 49.8)

        # Verify yield per strip calculations
        assert (round(image_to_yield_dict["20191012_150709_1049"] * 3 * 9, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191012_150714_1049"] * 3 * 9, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191012_162948_1035"] * 3 * 4, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191012_162955_1035"] * 3 * 4, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191013_164837_82_1057"] * 2 * 5, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191013_164845_96_1057"] * 2 * 5, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191014_150319_1054"] * 4 * 3, 2) == round(yearYield, 2))
        assert (round(image_to_yield_dict["20191014_150321_1054"] * 4 * 3, 2) == round(yearYield, 2))

"""
Test function: verify the contents of the pickle file
"""
def verify_pickle_contents():

    print("Starting verification test of pickle contents")

    # read and print out fips stored list
    completed_fips = []
    with open('json_store/labels/completed_fips.pkl', 'rb') as fp:
        completed_fips = pickle.load(fp)
    print(completed_fips)

    # read and output fips for labels
    sample_label_dict = {}
    sample_id = '3'
    with open('json_store/labels/' + sample_id + '.pkl', 'rb') as fp:
        sample_label_dict = pickle.load(fp)
    pprint.pprint(sample_label_dict)
