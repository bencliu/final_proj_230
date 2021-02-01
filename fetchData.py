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
from collections import Counter
from retrying import retry

#Define global variables
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'

#Print python dictionary data in readable way
def p(message="", data=None):
    print(message, json.dumps(data, indent=2))

"""
JSON Request Test Function
- Send get request to URL and print status
- Note: GET Req requires not data passed as param; POST Req generally carries client data (i.e. URL)
"""
def basic_req(apiKey):
    URL = "https://api.planet.com/data/v1"
    session = requests.Session() #Start session
    session.auth = (apiKey, "") #Authenticate
    res = session.get(URL) #Make GET request
    print(res.status_code)
    print("Body", res.json())

"""
Create Basic Search Filters:
- Stand-alone test function to look at STATs endpoint
- Returns "bucket" container holding images for each time period; count refers to number of images
- "Item" is a scene captured by a satellite: Props date acquired, geometry; "item-type" reps spacecraft captured
- "Asset" property derives from item source data
"""
def basic_search(apiKey):
    #Set up stats URL
    URL = "https://api.planet.com/data/v1"
    stats_url = "{}/stats".format(URL)

    #Filters: 1) Type filter prop; 2) Config; 3) Field name: Field on which to filter
    #Sensors and satellite to include in results
    item_types = ["PSScene3Band", "REOrthoTile"]

    #Filter for dates
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired", #Field to filter on: Date in which image taken
        "config": {
            "gte": "2020-01-01T00:00:00.000Z",  # "gte" -> Greater than or equal to
        }
    }

    #Create and send request
    request = {
        "item_types" : item_types,
        "interval": "month",
        "filter": date_filter
    }
    session = requests.Session()  # Start session
    session.auth = (apiKey, "")  # Authenticate
    res = session.post(stats_url, json=request)
    p(res.json())

"""
Helper function: Define GeoJSON filter for image search
- Note: GeoJSON.io or Planet Explorer API Config after drawing AOI
"""
def define_geometry():
    geojson_geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [-121.59290313720705, 37.93444993515032],
                [-121.27017974853516, 37.93444993515032],
                [-121.27017974853516, 38.065932950547484],
                [-121.59290313720705, 38.065932950547484],
                [-121.59290313720705, 37.93444993515032]
            ]
        ]
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
def search_image(api_key):
    print("Key", api_key)
    item_type = "PSScene4Band"
    combined_filter = define_filters()

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
"""
def extract_assets(api_key):
    print("Running extract_assets")
    item_type = "PSScene4Band"
    imageQ = search_image(api_key)

    #Extract image IDs
    image_ids = [feature['id'] for feature in imageQ.json()['features']]
    print("Number of Images:", len(image_ids))


    #Loop through images, process, and download TODO
    dateList = []
    locationList = []
    strips = []
    for index in range(15):
        assetResult, itemResult = get_item_asset(image_ids[index], item_type)
        date = itemResult.json()["properties"]["acquired"]
        x_orig = itemResult.json()["properties"]["origin_x"]
        y_orig = itemResult.json()["properties"]["origin_y"]
        strip = itemResult.json()["properties"]["strip_id"]
        p("Date", date)  # Print date in which scene was acquired
        p("X", x_orig)  # Print origin_x
        p("Y", y_orig)  # Print origin_y
        dateList.append(date)
        strips.append(strip)
        locationList.append((x_orig, y_orig))

    print("Datelist", dateList)
    print("LocList", locationList)
    stripDup = [k for k,v in Counter(strips).items() if v>1]
    print("StripDup", stripDup)


"""
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
    extract_assets(PLANET_API_KEY)







"""
[
                    -88.36441040039062,
                    41.71418635520256
                ],
                [
                    -88.25746536254883,
                    41.71418635520256
                ],
                [
                    -88.25746536254883,
                    41.770927598483546
                ],
                [
                    -88.36441040039062,
                    41.770927598483546
                ],
                [
                    -88.36441040039062,
                    41.71418635520256
                ]
"""

