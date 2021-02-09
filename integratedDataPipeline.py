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
import pickle

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
MAX_IMAGE_LENGTH = 0 #Global vars for image standardization
MAX_IMAGE_HEIGHT = 0 #Global vars for image standardization

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
Integrated Utility Func: Converting nested dictionary to image arrays
@Param: Nested dictionary of timesplit : [vector of {splitid : assetItemVec}]
@Return: Nested county image dictionary of timesplit : [vector of {splitid : image2dArrays}] 
"""
def integrated_download(nestedTimeSplitDict):
    nestedCountyArrays = {}
    for timeSplit, splitDictionary in nestedTimeSplitDict.items():
        for splitid, assetItemVec in splitDictionary.items():

            #Create image vector for parallezing
            assetVector = []
            for assetItemPair in assetItemVec:
                asset = assetItemPair[0]
                assetVector.append(asset)

            imageArray = parallelize(assetVector) #Will return tensor representation of images

            #Nested dictionary repackaging
            splits = {splitid: imageArray} #splitID : imageArrays
            nestedCountyArrays[timeSplit].append(splits) #timeSplit may have multiple splits
    return nestedCountyArrays

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
        countyImages = integrated_download(nestedTimeSplitStruct)

        #Store county images in pre-loadable file via pickle
        fullPath = 'images/'+fipCode+'.pickle'
        with open(fullPath, 'wb') as handle:
            pickle.dump(countyImages, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Helper to unserialize pickle data object
def retrievePickle(path):
    # Load data (deserialize)
    with open(path, 'rb') as handle:
        unserialized_data = pickle.load(handle)
        return unserialized_data

if __name__ == "__main__":
    print("Starting Integrted Data Pipeline")