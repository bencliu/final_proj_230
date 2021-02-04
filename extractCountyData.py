""""
File Description: Collection of functions for extracting county-level metrics
"""

# Imports
import geojson
import re
import csv
from typing import Any, Dict, Tuple, List, Callable

"""
Function: Extracts the county FIPS code and polygon coordinates into a dictionary
@Param: filename of GeoJSON as string
@Returns: dict[County FIP: List[List[Coordinates]]]
"""
def read_county_GeoJSON(filename: str):

    # Read in GeoJSON file
    with open(filename) as f:
        gj = geojson.load(f)

    # Create Dictionary of FIP codes and coordinates
    county_polygons = {}
    for i in range(0, len(gj['features'])):

        # Extract FIPS code from Feature Collection Descriptions
        feature_description = gj['features'][i]['properties']['Description']
        search_string = 'COUNTY num:<\/b>(.*)<\/div>'
        result = re.search(search_string, feature_description)
        county_FIP = int(result.group(1))

        # Add FIPS
        name = gj['features'][i]['properties']['Name']
        # county_polygons[county_FIP] = name

        # Add Polygon Coordinates 
        [county_geometry] = gj['features'][i]['geometry']['coordinates']
        county_polygons[county_FIP] = county_geometry

    # dictionary of {FIP: list of list coordinates}
    return county_polygons

"""
Function: Extracts county yield truth data into a nested dictionary
Note: Empty FIP codes correspond to "OTHER (COMBINED) COUNTIES"
@Param: filename of county yields csv as string
@Returns: dict[County FIP: dict[year, yield]]
"""
def read_county_truth(filename: str): 

    county_truth_yields = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter = ',', skipinitialspace = True)
        next(reader)
        for row in reader:

            # extract county metrics
            county_yield = float(row[-2])
            year = int(row[1])
            county_name = row[9]

            # eliminate empty county FIPS
            if not row[10]:
                continue
            county_FIP = int(row[10])

            # populate nested dictionary with annual yields for each county
            if county_FIP not in county_truth_yields:
                county_truth_yields[county_FIP] = {year: county_yield}
            else:
                county_truth_yields[county_FIP][year] = county_yield
    
    
    # return nested dictionary of dict[County FIP: dict[year, yield]]
    return county_truth_yields

if __name__ == "__main__":
    
    # Extract county polygons from GeoJSON
    county_polygons = read_county_GeoJSON("Illinois_counties.geojson")

    # Extract county truth yields from csv
    county_truth_yields = read_county_truth("Illinois_Truth_Data.csv")




