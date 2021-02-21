""""
File Description: Key functions for pulling data from planet labs and building
basic tensors with human-engineered vegetation features

API Key: b99bfe8b97d54205bccad513987bbc02

Notes:
- Filters: Limit search results based on date, geography, metadata
- Search: Get items or generate stats
"""

#Import relevant packages
import numpy as np
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from collections import Counter, OrderedDict
import dateutil.parser
import datetime
from retrying import retry
from planet import api
from typing import Any, Dict, Tuple, List, Callable

#Define global variables
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'

#Print python dictionary data in readable way
def p(message="", data=None):
    print(message, json.dumps(data, indent=2))

"""
Helper function: Define GeoJSON filter for image search
- Note: GeoJSON.io or Planet Explorer API Config after drawing AOI
"""
def define_geometry():
    geojson_geometry = {
        "type": "Polygon",
        "coordinates": [
          [
            [
              -89.74731445312499,
              39.760783168525926
            ],
            [
              -89.69856262207031,
              39.729633617508426
            ],
            [
              -89.63642120361328,
              39.73729032077588
            ],
            [
              -89.64225769042969,
              39.77424175134451
            ],
            [
              -89.70268249511719,
              39.78321267821705
            ],
            [
              -89.74731445312499,
              39.760783168525926
            ]
          ]
        ]

    }
    return geojson_geometry

"""
Helper function: Define GeoJSON filter for image search based on county polygon
@Param: County FIP code, County Polygons generated nested list
@Returns: geometry filter for county
Note: This was changed due to the code design of the integrated data pipeline; It's inefficient to
      pass in a dictionary at every call to this function 
"""
def define_county_geometry(coordinates: List[List[float]]):
    geojson_geometry = {
        "type": "Polygon",
        "coordinates": coordinates
    }
    return geojson_geometry

"""
Helper function: Define search filters for query
- Geom filter: Geojson filter
- Date Range filter: Dates and increments 
- Cloud Cover Filter: Eliminates outlier cloud images
"""
def define_filters():
    # get images that overlap with our AOI
    geojson = define_geometry()
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
            "gte": "2016-08-31T00:00:00.000Z",
            "lte": "2016-11-01T00:00:00.000Z"
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

"""
Helper function: Search Image
- Integrates combined filter from define_filer() helper function
- Sends and receives search request result 
"""
def search_image(api_key, combined_filter):
    print("Key", api_key)
    item_type = "PSScene4Band"

    #API Req Object
    search_request = {
        "item_types": [item_type],
        "filter": combined_filter
    }

    #Post req to quicksearch: Returns ID's of Items as one field
    search_result = requests.post('https://api.planet.com/data/v1/quick-search',
        auth=HTTPBasicAuth(api_key, ''),
        json=search_request)

    return search_result

"""
Function: Extract asset and image object given ID
"""
def get_item_asset(id, item_type):
    id0_asset_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, id)
    id0_item_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}'.format(item_type, id)

    # Returns JSON metadata for assets in this ID. Learn more: planet.com/docs/reference/data-api/items-assets/#asset
    assetResult = \
        requests.get(
            id0_asset_url,
            auth=HTTPBasicAuth(PLANET_API_KEY, '')
        )

    # Returns JSON metadata for item corresponding to this ID
    itemResult = \
        requests.get(
            id0_item_url,
            auth=HTTPBasicAuth(PLANET_API_KEY, '')
        )

    return assetResult, itemResult


"""
Helper function: Extract asset from search result post request
- Accesses image IDs and sends get request to return image metadata
- Note: itemResult contains metadata about image; assetResult contains actual image for activation

@Param: API Key
@Return: Dictionary mapping datetime objects to tuple (itemResult, assetResult)
"""
def extract_assets(api_key, combined_filter):
    print("Running extract_assets")
    item_type = "PSScene4Band"
    imageQ = search_image(api_key, combined_filter)

    #Extract image IDs
    image_ids = [feature['id'] for feature in imageQ.json()['features']]
    print("Number of Images:", len(image_ids))


    #Loop through images and gather relevant activation and metadata
    dateList = []
    strips = []
    asset_item_dictionary = {}
    for index in range(len(image_ids)):
        assetResult, itemResult = get_item_asset(image_ids[index], item_type)
        date = itemResult.json()["properties"]["acquired"]
        strip = itemResult.json()["properties"]["strip_id"]
        dateTimeObj = dateutil.parser.isoparse(date)
        asset_item_dictionary[dateTimeObj] = (assetResult, itemResult)
        dateList.append(date)
        strips.append(strip)

    #stripDup = [k for k,v in Counter(strips).items() if v>1] Detect strip duplicates
    return asset_item_dictionary #Vector of items and corresponding assets

"""
**NOTE: WRAPPER FUNC FOR ORDERS.PY**
Helper function: Extract items from search result post request
- Accesses image IDs and sends get request to return image metadata

@Param: API Key
@Return: Dictionary mapping datetime objects to itemResult
"""
def extract_items(api_key, combined_filter):
    print("Running extract_assets")
    item_type = "PSScene4Band"
    imageQ = search_image(api_key, combined_filter)

    #Extract image IDs
    image_ids = [feature['id'] for feature in imageQ.json()['features']]
    print("Number of Images:", len(image_ids))

    #Loop through images and gather relevant activation and metadata
    item_dictionary = {}
    for index in range(len(image_ids)):
        assetResult, itemResult = get_item_asset(image_ids[index], item_type)
        date = itemResult.json()["properties"]["acquired"]
        dateTimeObj = dateutil.parser.isoparse(date)
        item_dictionary[dateTimeObj] = itemResult

    return item_dictionary #Dictionary of dateTimeObjects with ItemResults

"""
Helper function: Separate time series vectors
@Param: Dictionary of datetimeObjects : (assetResult, itemResult) tuples
@Return: Dictionary: [Time: Vector of (assetResult, itemResult) tuples]

-Sort time objects 
-Loop through and separate by changing dates
"""
def split_into_time_series(assets_items):
    #Sort dictionary
    print("STARTING Split Times")
    sorted_time_dict = OrderedDict(sorted(assets_items.items()))
    timeSliceDict = OrderedDict()

    #Loop through and separate time slices (Based on day in this implementation)
    #Note: TimeBuffer value of 3 means time separation at day level,, 2 is month level
    for key, val in sorted_time_dict.items():
        # Separation by day as time slice
        timeSliceIdentifier = datetime.datetime(key.year, key.month, key.day)
        if timeSliceIdentifier not in timeSliceDict:
            timeSliceDict[timeSliceIdentifier] = []
        timeSliceDict[timeSliceIdentifier].append(val)

    print("FINISHING Split Time Series")
    return timeSliceDict

"""
Helper function: Restructures data by collective landscape strips
Note: singleItem boolean refers to whether helper is processing tuple or itemResult single element
@Param: Takes in Dictionary of datetimeObjects: (assetResult, itemResult) vector or (itemResult) vector
@Return: Dictionary of dateTimeObjects: [Dictionary of stripIDs: (assetResult, itemResult) vector]
"""
def split_by_strip(timeSliceDict, singleItem=False):
    split_time_dict = OrderedDict()

    #Loop through each time split
    for time_split, tupleVec in timeSliceDict.items():
        strip_dictionary = separate_by_strips(tupleVec, singleItem=singleItem) #Construct Dictionary
        split_time_dict[time_split] = strip_dictionary #Update new dict

    return split_time_dict

"""
Helper function: Separates vector of asset, item tuples into dictionary of splits
Note: singleItem boolean refers to whether helper is processing tuple or itemResult single element
@Param: item_asset_vector (Grouped by time slice)
@Return: Dictionary mapping stripIDs to item_asset_vector
"""
def separate_by_strips(item_asset_vec, singleItem=False):
    strip_dict = {}

    #Loop through pairs
    for item_asset_pair in item_asset_vec:
        stripid = item_asset_pair.json()["properties"]["strip_id"] \
            if singleItem else item_asset_pair[1].json()["properties"]["strip_id"]
        if stripid not in strip_dict: #Create new vector if new id
            strip_dict[stripid] = []
        strip_dict[stripid].append(item_asset_pair) #Add pair to vector corresponding to id
    return OrderedDict(sorted(strip_dict.items()))

"""
Test Function: Tests that timesplit/splitID separation is working
@Param: split_time_dict: Dictionary of time_splits : [Dictionary of stripIDs : asset_item_vec]
@Return: True/False depending on test success
"""
def test_time_split(split_time_dict):
    print("Starting test function")
    #Look at each asset_item pair and see if the date and stripID matches with categorization
    for time_split, strip_dict in split_time_dict.items():
        for strip_id, asset_item_vector in strip_dict.items():
            for asset_item in asset_item_vector:
                item = asset_item[1]
                
                #Date parsing 
                itemDate = item.json()["properties"]["acquired"]
                itemDate = dateutil.parser.isoparse(itemDate)
                itemDateTime = datetime.datetime(itemDate.year, itemDate.month, itemDate.day)
                
                rightTime = itemDateTime == time_split #Check time split matches
                rightStrip = (strip_id == item.json()["properties"]["strip_id"]) #Check stripid matches
                if not rightTime or not rightStrip:
                    return False 
    return True

""""
Helper function: Load JSON objects into file for later use
- Used for image and assets pre-activation 
"""
def write_to_json(data, outfile):
    with open(outfile, 'w') as outfile:
        json.dump(data, outfile)

"""
Helper function: Load JSON objects into file for later use
- Read JSON objects to load
"""
def read_json(infile):
    with open(infile) as json_file:
        data = json.load(json_file)
        return data

if __name__ == "__main__":
    #b, g, r, n = simple_image_process("../ArchiveData/planet_sample1.tif")
    #construct_tensors(b, g, r, n)
    #PLANET_API_KEY = os.getenv('PL_API_KEY')
    combined_filter = define_filters()
    item_asset_dict = extract_assets(PLANET_API_KEY, combined_filter)
    timeSliceDict = split_into_time_series(item_asset_dict)
    finalStruct = split_by_strip(timeSliceDict)
    print(test_time_split(finalStruct))





