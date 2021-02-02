""""
File Description: Basic Test Functions working with Planet-Labs API
- Send basic API request
- Send basic API req to stats endpoint

API Key: b99bfe8b97d54205bccad513987bbc02

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
