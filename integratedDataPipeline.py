""""
File Description:
- Integrated utility functions for the data pipeline

API Key: b99bfe8b97d54205bccad513987bbc02

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
from typing import Any, Dict, Tuple, List, Callable

#Import county extraction functions
from extractCountyData import read_county_GeoJSON

#Import download_asset functions
from downAssets import extract_images, act_download_asset, parallelize, download_file

#Import fetchData functions
from fetchData import define_county_geometry, define_filters #Filter Functions
from fetchData import search_image, get_item_asset, extract_assets #Search Functions
from fetchData import split_into_time_series, split_by_strip, separate_by_strips


#Define global variables
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'

"""
Integrated Utility Func: Filter, Search, Obtain Images
@Param: combined search filter
@Return: Nested dictionary timeslices => stripids => images 
"""
def integrated_search(combined_filter):
    imagesDictionary = extract_assets(PLANET_API_KEY, combined_filter)
    timeSliceDictionary = split_into_time_series(imagesDictionary)
    timeSplitNested = split_by_strip(timeSliceDictionary)
    return timeSplitNested

"""
Integrated Utility Func: Downloading Images 
@Param: vector of tuples (assetResult, itemResult)
@Action: Downloads images into nested folder structure
Note: this is a dumb function, just a symbolic representation for now 
"""
def integrated_download(nestedTimeSplitDict):
    for timeSplit, splitDictionary in nestedTimeSplitDict.items():
        for splitid, assetItemVec in splitDictionary.items():
            pass
            #TODO: Pass to parallelize, download_file series for nested folder structure

"""
TODO: Need to define our scope of filter params
"""
def combined_filter(geoFilter):
    combined_filter = {}
    return combined_filter

"""
Integrated Pipeline Func: Process Counties, Download All Information
@Param: Receives dictionary of FIP Codes => Coordinates of AOIs
@Action: Obtain/Organize all images, Download them into nested folder structure 
"""
def integrated_pipeline(county_dictionary, root_path='../countyImages'):
    for fipCode, coordinates in county_dictionary.items():
        #Build filters and search for county images
        countyCoordinates = county_dictionary[fipCode]
        geoFilter = define_county_geometry(countyCoordinates)
        searchFilter = combined_filter(geoFilter)

        #Integrated search and download of county images
        nestedTimeSplitStruct = integrated_search(searchFilter)
        integrated_download(nestedTimeSplitStruct)



"""
Notes: Needed functions
- Well defined file structures 
- Integrated pipeline for image processing 
"""

if __name__ == "__main__":
    print("Starting Integrted Data Pipeline")