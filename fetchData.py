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
import math
import rasterio
import matplotlib.pyplot as plt
from requests.auth import HTTPBasicAuth

#Define global variables
PLANET_API_KEY = os.getenv('PL_API_KEY')

#Print python dictionary data in readable way
def p(data):
    print(json.dumps(data, indent=2))

#JSON request test function
def basic_req(apiKey):
    URL = "https://api.planet.com/data/v1"
    session = requests.Session() #Start session
    session.auth = (apiKey, "") #Authenticate
    res = session.get(URL) #Make GET request
    print(res.status_code)
    print("Body", res.json())

#Function to create basic search filters
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
            "lte": "2016-09-01T00:00:00.000Z"
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

def search_image():
    item_type = "PSScene4Band"
    combined_filter = define_filters()

    #API Req Object
    search_request = {
        "item_types": [item_type],
        "filter": combined_filter
    }

    #Send Req object through post request quick search
    search_result = requests.post('https://api.planet.com/data/v1/quick-search',
        auth=HTTPBasicAuth(PLANET_API_KEY, ''),
        json=search_request)

    return search_result

def activate_download_image():
    item_type = "PSScene4Band"
    imageQ = search_image()

    #Extract image IDs
    image_ids = [feature['id'] for feature in imageQ.json()['features']]

    #First Image ID
    first_image_id = image_ids[0]
    id0_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, first_image_id)

    # Returns JSON metadata for assets in this ID. Learn more: planet.com/docs/reference/data-api/items-assets/#asset
    result = \
        requests.get(
            id0_url,
            auth=HTTPBasicAuth(PLANET_API_KEY, '')
        )



    #Activate asset for download

    # Parse out useful links
    links = result.json()[u"analytic"]["_links"]
    self_link = links["_self"]
    activation_link = links["activate"]

    # Request activation of the 'analytic' asset:
    activate_result = \
        requests.get(
            activation_link,
            auth=HTTPBasicAuth(PLANET_API_KEY, '')
        )


    # Wait until activation complete, then print download link
    while True:
        activation_status_result = \
            requests.get(
                self_link,
                auth=HTTPBasicAuth(PLANET_API_KEY, '')
            )
        if activation_status_result.json()["status"] == 'active':
            download_link = activation_status_result.json()["location"]
            print(download_link)
            break

    ##TODO: Learn how to download image to folder automatically via link

#Image processing of image into tensor
def simple_image_process():
    #Obtain image
    imagePath = "../ArchiveData/planet_sample1.tif"
    sat_data = rasterio.open(imagePath)

    #Image dimensions
    width = sat_data.bounds.right - sat_data.bounds.left
    height = sat_data.bounds.top - sat_data.bounds.bottom

    # Upper left pixel
    row_min = 0
    col_min = 0

    # Lower right pixel.  Rows and columns are zero indexing.
    row_max = sat_data.height - 1
    col_max = sat_data.width - 1

    # Conversion to numpy arrays
    b, g, r, n = sat_data.read()
    print(g.shape)
    one_hot = np.zeros((b.shape[1], 1))
    print(one_hot.shape)
    nonzero_g = np.matmul(g, one_hot)
    print(nonzero_g.shape)
    print("testing")

    #nonzero_g = [elem in nonzero_g if elem != 0]
    #print("Blue", nonzero_g.T)

if __name__ == "__main__":
    #basic_req(PLANET_API_KEY)
    #activate_download_image()
    simple_image_process()



