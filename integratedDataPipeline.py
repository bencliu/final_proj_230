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

            #Create image vector for parallelizing
            assetVector = []
            for assetItemPair in assetItemVec:
                asset = assetItemPair[0]
                assetVector.append(asset)

            imageArray = parallelize(assetVector) #Will return tensor representation of images

            #Nested dictionary repackaging
            splits = {splitid: imageArray} #splitID : imageArrays
            if not nestedCountyArrays[timeSplit.strftime("%m/%d/%Y")]:
                nestedCountyArrays[timeSplit.strftime("%m/%d/%Y")] = [] #timeSplit may have multiple splits
            nestedCountyArrays[timeSplit.strftime("%m/%d/%Y")].append(splits)

    return nestedCountyArrays

"""
Note: These are just sample filters, we can change them for our baseline
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
            "gte": "2018-09-02T00:00:00.000Z",
            "lte": "2018-09-02T20:00:00.000Z"
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
        "CountyLevel: Start search for images"
        nestedTimeSplitStruct = integrated_search(searchFilter)
        countyImages = integrated_download(nestedTimeSplitStruct)

        #Store county images in pre-loadable file via pickle
        print("Starting pickle storing process for county")
        fullPath = 'images/'+str(fipCode)+'.pickle'
        with open(fullPath, 'wb') as handle:
            pickle.dump(countyImages, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Finished pickle storing process for county")
        break


#Helper to unserialize pickle data object
def retrievePickle(path):
    # Load data (deserialize)
    with open(path, 'rb') as handle:
        unserialized_data = pickle.load(handle)
        return unserialized_data

"""
Integrated pipeline tester:
- Test one county
- Notes: Works, but 18 images is stored on 2.64 GB. We'll have to use AWS for 
- these operations
"""
def testOneCounty():
    countyDict = read_county_GeoJSON(filename='json_store/Illinois_counties.geojson')
    integrated_pipeline(countyDict)

"""
Unserialize and Analyze Pickled Objects
"""
def testPickledObject():
    nestedCountyArray = retrievePickle("./images/1.pickle")
    for timeSplits, splitArray in nestedCountyArray.items():
        for splitDict in splitArray:
            for splitid, split in splitDict.items():
                print(split)

if __name__ == "__main__":
    print("Starting Integrated Data Pipeline")
    testPickledObject()
