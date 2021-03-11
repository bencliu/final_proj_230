""""
File Description: Collection of functions for extracting county-level metrics
"""

# Imports
import geojson
import re
import csv
from typing import Any, Dict, Tuple, List, Callable
from fetchData import define_county_geometry
from collections import Sequence
from itertools import chain, count
import downAssets
import fetchData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import pickle
import locale
import matplotlib.pyplot as plot
import pprint
import unidecode
import os

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
        county_geometry = gj['features'][i]['geometry']['coordinates']
        county_polygons[county_FIP] = county_geometry

    pprint.pprint(county_polygons)

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

"""
Function: Extract regional yield truth data into nested dictionary for brazil from raw paper
source: https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction/blob/master/code/static_data_files/brazil_yields_standardized.csv
@Param: n/a
@Return dict[Region name: dict[year, yield] 
"""
def read_brazil_regional_truth():

    # read in filtered soybean regions
    regions = []
    with open('json_store/brazil_soybeans_filtered_regions.txt', 'r') as file:
        for line in file:
            region = line.split("-")[0]
            regions.append(region)
    file.close()

    regional_truth_yields = {}
    with open('json_store/brazil_yields_standardized.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        next(reader)
        for row in reader:

            # extract regional metrics
            region = str(row[1])
            year = int(row[2])
            regional_yield = float(row[-1])

            # only keep soybean filtered regions
            if region not in regions:
                continue

            # populate nested dictionary with annual yields for each county
            if region not in regional_truth_yields:
                regional_truth_yields[region] = {year: regional_yield}
            else:
                regional_truth_yields[region][year] = regional_yield
    print(regional_truth_yields)
    file.close()

"""
Function: Extract county areas
"""
def read_illinois_county_area():

    # extract excel contents to lists
    county_areas = {}
    with open('json_store/Illinois_County_Areas.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        next(reader)
        for row in reader:
            fipCode = int(row[1])
            areaString = row[-2]
            extractedString = re.search('mi(.*)km2', areaString)
            isolatedString = extractedString.group(1)[1:]
            locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )
            rawArea = locale.atoi(isolatedString)
            area = rawArea * 1000000; # convert from km^2 to m^2
            county_areas[fipCode] = area
    file.close()

    with open('data/archive/county_areas.p', 'wb') as fp:
        pickle.dump(county_areas, fp)

"""
Function: Extracts truth from municipal agricultural production website (SIDRA)
Source: https://sidra.ibge.gov.br/tabela/1612
@Params: None
@Return dict[Region name: dict[year, yield] 
"""
def read_brazil_regional_truth_v2():

    # extract excel contents to lists
    xl_file = pd.read_excel('json_store/Brazil_Truth.xlsx', engine='openpyxl', skiprows=4, skipfooter=1)
    cols = xl_file.columns.tolist()
    mesoregions = xl_file[cols[0]].tolist()
    yield_2017 = xl_file[cols[1]].tolist()
    yield_2018 = xl_file[cols[2]].tolist()
    yield_2019 = xl_file[cols[3]].tolist()

    # populate nested dictionary with annual yields for each mesoregion
    tobush = 1/67.25
    mesoregional_truth_yields = {}
    for i in range(len(mesoregions)):
        mesoregion = mesoregions[i].split(" (")[0]
        if isinstance(yield_2017[i], str) or isinstance(yield_2018[i], str) or isinstance(yield_2019[i], str):
            continue # skip if mesoregion is missing contents
        if mesoregions[i] not in mesoregional_truth_yields:
            mesoregional_truth_yields[mesoregion] = {2017: yield_2017[i] * tobush,
                                                         2018: yield_2018[i] * tobush,
                                                         2019: yield_2019[i] * tobush}
        else:
            mesoregional_truth_yields[mesoregion][2017] = yield_2017[i] * tobush
            mesoregional_truth_yields[mesoregion][2018] = yield_2018[i] * tobush
            mesoregional_truth_yields[mesoregion][2019] = yield_2019[i] * tobush
    print(mesoregional_truth_yields)

"""
Function: Extracts truth from municipal agricultural production website (SIDRA)
Source: https://sidra.ibge.gov.br/tabela/1612
@Params: None
@Return dict[Region name: dict[year, yield] 
"""
def read_brazil_regional_truth_v3():

    # extract excel contents to lists
    xl_file = pd.read_excel('json_store/brazil_data/brazil_muni_yields.xlsx', engine='openpyxl', skiprows=4, skipfooter=1)
    cols = xl_file.columns.tolist()
    raw_munis = xl_file[cols[0]].tolist()
    raw17 = xl_file[cols[1]].tolist()
    raw18 = xl_file[cols[2]].tolist()
    raw19 = xl_file[cols[3]].tolist()

    # extract contents for mato grosso
    totals = [] # for debug purposes
    tobush = 1 / 67.25
    filtered_muni_truth_yields ={}
    filtered_munis = []
    for i in range(len(raw_munis)):
        if raw_munis[i][raw_munis[i].find("(") + 1:raw_munis[i].find(")")] == "MT":
            if isinstance(raw17[i], str) or isinstance(raw18[i], str) or isinstance(raw19[i], str):
                continue  # skip if municipality if missing yields
            totals.append(raw17[i] + raw18[i] + raw19[i])  # for debug purposes
            if raw17[i]+raw18[i]+raw19[i] <= 10180:
                continue # filter out
            muni = raw_munis[i].split(" (")[0].upper()
            muni = unidecode.unidecode(muni) # remove accents and special characters
            filtered_muni_truth_yields[muni] = {2017: raw17[i] * tobush,
                                                     2018: raw18[i] * tobush,
                                                     2019: raw19[i] * tobush}
            filtered_munis.append(muni)

    """plot.hist(totals, density=True, bins=100)
    plot.show()"""
    pprint.pprint(filtered_muni_truth_yields)
    print(len(filtered_muni_truth_yields.keys()))

    # Write to pickle file
    with open('json_store/brazil_data/filtered_brazil_yields.p', 'wb') as fp:
        pickle.dump(filtered_muni_truth_yields, fp)
    with open('json_store/brazil_data/filtered_brazil_munis.p', 'wb') as fp:
        pickle.dump(filtered_munis, fp)

"""
Function: Extract GEOJSON of counties
Note: Output pickle file of dict[Muni Name: List[List[Coordinates]]]
"""

def extract_brazil_geojson():

    # extract filtered list of munis
    with open("json_store/brazil_data/filtered_brazil_munis.p", 'rb') as fp:
        filtered_munis = pickle.load(fp)

    # Read in GeoJSON file
    filename = "/Users/christopheryu/Desktop/mygeodata/processed/05_malhamunicipal1991_7.json"
    with open(filename) as f:
        gj = geojson.load(f)

    # Create Dictionary of FIP codes and coordinates
    muni_polygons = {}
    count = 0
    debug = []
    for i in range(0, len(gj['features'])):

        # Extract FIPS code from Feature Collection Descriptions
        properties = gj['features'][i]['properties']
        muni_name = gj['features'][i]['properties']['NOMEMUNICP']
        if muni_name in filtered_munis:
            debug.append(muni_name)
            count += 1
            muni_polygons[muni_name] = gj['features'][i]['geometry']['coordinates']

    pprint.pprint(muni_polygons)
    # write geojson to pickle file
    with open('json_store/brazil_data/filtered_brazil_geojson_7.p', 'wb') as fp:
        pickle.dump(muni_polygons, fp)

    print(count)
    print(len(filtered_munis))

"""
Function: Assemble Master Label Dictionary
"""
def assemble_master_label_dictionary():

    # combine pickle files
    master_label_dict = {}
    filepath = "json_store/labels_v3"
    for filename in os.listdir(filepath):
        if filename.endswith(".pkl"):
            with open('json_store/labels_v3/' + str(filename), 'rb') as fp:
                dict = pickle.load(fp)
                master_label_dict = {**master_label_dict, **dict}

    # write to pickle file
    with open('data/archive/master_label_dict_v3.p', 'wb') as fp:
        pickle.dump(master_label_dict, fp)

"""
Function: This function bins the truth data based on the classification split
@Params: truth_data of shape (# of examples, 1), binned ranges np.array
@Returns: array of shape (# of examples, 1) binned into the given number of classes
"""
def bin_truth_data(truth_data, bins_array):
    binned_truth = np.digitize(truth_data, bins_array)
    return binned_truth

"""
Function: This method reads individual county .pkl dict{key: id, values: yield/county}
Note: Only run this funcntion once all the local pickle files have been generated
@Params: Filepath to processed FIPs pickle file (output of obtain_crop_labels), Number of desired classes
@Return: None
"""
def export_classification_labels(completed_fips_list_file_path: str, num_classes: int):

    # read in list of FIPS
    print("----------Start process to create labels----------")
    completed_fips = []
    with open(completed_fips_list_file_path, 'rb') as fp:
        completed_fips = pickle.load(fp)

    # read in all county label dictionary files
    master_label_dict = {}
    for fipCodeYear in completed_fips:
        county_label_dict = {}
        with open('json_store/labels_v2/' + str(fipCodeYear) + '.pkl', 'rb') as fp:
            county_label_dict = pickle.load(fp)
            master_label_dict = {**master_label_dict, **county_label_dict}
    print("----------Finished reading in .pkl files----------")

    """# Perform the quantile cut
    values = list(master_label_dict.values()) # extract values from
    np.random.seed(42)
    test_list_rnd = np.array(values) + np.random.random(len(values))  # add noise to data
    test_series = pd.Series(test_list_rnd, name='value_rank')
    split_df = pd.qcut(test_series, q=num_classes, retbins=True, labels=False)
    # split_df = pd.qcut(yields, q=num_classes)
    bins = split_df[1]
    print("----------Classification Quantile Cut Complete----------")

    # bin the original dictionary
    for key, val in master_label_dict.items():
        [cropLabel] = bin_truth_data(np.array([val]), bins)
        master_label_dict[key] = cropLabel  # convert to label"""

    # Write to pickle file
    with open('json_store/labels_v2/master_label_dict.pkl', 'wb') as fp:
        pickle.dump(master_label_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("----------Classification Label Dictionary Complete----------")
    print("Navigate to json_store/labels_v2/master_label_dict_binned.pkl for completed product")

if __name__ == "__main__":
    
    # Extract county polygons from GeoJSON
    # county_polygons = read_county_GeoJSON("json_store/Illinois_counties.geojson")

    # Extract county truth yields from csv
    # county_truth_yields = read_county_truth("json_store/Illinois_Soybeans_Truth_Data.csv")

    # Extract brazil regional yields
    # read_brazil_regional_truth()
    # read_brazil_regional_truth_v3()

    # Extract brazil geojson
    extract_brazil_geojson()

    # extract illinois area dictionary
    # read_illinois_county_area()

    # Obtain classification labels from truth data
    # export_classification_labels(completed_fips_list_file_path = 'json_store/labels_v2/completed_fips.pkl', num_classes = 10)
    # assemble_master_label_dictionary()

    """with open('data/archive/master_label_dict_v3.p', 'rb') as fp:
        labels = pickle.load(fp)
    with open('data/aws_file_dict_vUpdate2.p', 'rb') as fp:
        paths = pickle.load(fp)

    pprint.pprint(labels)"""


"""
Archive Functions ===============================================================
"""

"""
Test function: Yield extraction
Randomly selected test examples from USDA Quickstat database
@Params: None
@Return: Nonte
"""
def test_truth_yield():

    # generate county truth yields
    county_truth_yields = read_county_truth("json_store/Illinois_Soybeans_Truth_Data.csv")

    # sample test cases
    assert(county_truth_yields[115][2019] == 63.8)
    assert(county_truth_yields[37][2016] == 67.5)
    assert(county_truth_yields[197][2011] == 51.1)
    assert(county_truth_yields[65][2007] == 25)
    assert(county_truth_yields[63][2004] == 49)

    # Print completion statement
    print("Test for county yield generation is complete")

"""
Test function: GeoJSON extraction
@Params: None
@Return: None
"""
def test_county_GeoJSON():

    # generate county_polygons
    county_polygons = read_county_GeoJSON("json_store/Illinois_counties.geojson")

    # sample test cases
    assert(county_polygons[1]  == [ [ [ -91.42192, 39.93308 ], [ -91.42073, 39.93123 ], [ -91.41907, 39.92574 ],
                                    [ -91.41889, 39.92141 ], [ -91.41988, 39.91653 ], [ -91.4218, 39.91398 ], [ -91.42371, 39.91214 ], [ -91.43441, 39.90242 ], [ -91.44351, 39.89358 ], [ -91.44711, 39.882 ], [ -91.44638, 39.87039 ], [ -91.43733, 39.84946 ], [ -91.43653, 39.84708 ], [ -91.42149, 39.83372 ], [ -91.40509, 39.82554 ], [ -91.39314, 39.81913 ], [ -91.37515, 39.80886 ], [ -91.36848, 39.80102 ], [ -91.3618, 39.78873 ], [ -91.36347, 39.78242 ], [ -91.36569, 39.77491 ], [ -91.36591, 39.76496 ], [ -91.36512, 39.75872 ], [ -91.35264, 39.75858 ], [ -91.33798, 39.75856 ], [ -91.32164, 39.75754 ], [ -91.30514, 39.75799 ], [ -91.2646, 39.75723 ], [ -91.21991, 39.75702 ], [ -91.19039, 39.75693 ], [ -91.14878, 39.75705 ], [ -91.14806, 39.75706 ], [ -91.09738, 39.7572 ], [ -91.08186, 39.75715 ], [ -91.0581, 39.75715 ], [ -91.041, 39.75712 ], [ -91.0291, 39.75699 ], [ -91.01508, 39.757 ], [ -91.0092, 39.75701 ], [ -90.99344, 39.75687 ], [ -90.97512, 39.75687 ], [ -90.96369, 39.75698 ], [ -90.95522, 39.75707 ], [ -90.94311, 39.75713 ], [ -90.92963, 39.7572 ], [ -90.91638, 39.78915 ], [ -90.91643, 39.79652 ], [ -90.91626, 39.80433 ], [ -90.9165, 39.83038 ], [ -90.91661, 39.84507 ], [ -90.91666, 39.8522 ], [ -90.91674, 39.85621 ], [ -90.91686, 39.86913 ], [ -90.91677, 39.882 ], [ -90.91684, 39.88773 ], [ -90.91677, 39.89519 ], [ -90.9168, 39.90362 ], [ -90.91663, 39.91146 ], [ -90.91607, 39.91765 ], [ -90.91453, 39.92919 ], [ -90.91427, 39.9383 ], [ -90.91453, 39.95163 ], [ -90.91447, 39.97384 ], [ -90.91445, 39.98856 ], [ -90.91443, 40.00005 ], [ -90.91445, 40.0247 ], [ -90.91444, 40.02668 ], [ -90.91425, 40.03541 ], [ -90.91407, 40.04509 ], [ -90.91405, 40.0602 ], [ -90.9136, 40.08396 ], [ -90.91359, 40.08771 ], [ -90.91355, 40.08845 ], [ -90.9136, 40.09026 ], [ -90.91356, 40.09361 ], [ -90.9135, 40.09793 ], [ -90.91362, 40.10445 ], [ -90.91358, 40.10713 ], [ -90.91325, 40.11946 ], [ -90.9132, 40.12219 ], [ -90.91315, 40.125 ], [ -90.9131, 40.12745 ], [ -90.91297, 40.13314 ], [ -90.91296, 40.13542 ], [ -90.91291, 40.14008 ], [ -90.9129, 40.14097 ], [ -90.91289, 40.14207 ], [ -90.91289, 40.14339 ], [ -90.91286, 40.14552 ], [ -90.9128, 40.14825 ], [ -90.91277, 40.1489 ], [ -90.9126, 40.15798 ], [ -90.91258, 40.1603 ], [ -90.91254, 40.16297 ], [ -90.91248, 40.16603 ], [ -90.91246, 40.16888 ], [ -90.91238, 40.17387 ], [ -90.91236, 40.17544 ], [ -90.91226, 40.17992 ], [ -90.91213, 40.18404 ], [ -90.91211, 40.18599 ], [ -90.91207, 40.1875 ], [ -90.912, 40.19212 ], [ -90.91197, 40.19309 ], [ -90.91556, 40.19313 ], [ -90.96548, 40.19361 ], [ -91.0003, 40.19412 ], [ -91.02742, 40.19447 ], [ -91.06609, 40.1949 ], [ -91.0956, 40.19516 ], [ -91.12627, 40.19543 ], [ -91.1453, 40.19554 ], [ -91.16657, 40.19597 ], [ -91.18173, 40.19621 ], [ -91.20241, 40.19669 ], [ -91.21166, 40.19682 ], [ -91.22439, 40.19696 ], [ -91.24309, 40.19712 ], [ -91.25883, 40.1973 ], [ -91.29456, 40.19797 ], [ -91.31376, 40.19814 ], [ -91.34057, 40.19857 ], [ -91.35341, 40.19876 ], [ -91.37749, 40.19915 ], [ -91.4035, 40.19953 ], [ -91.42512, 40.1997 ], [ -91.45392, 40.19981 ], [ -91.45512, 40.19982 ], [ -91.48091, 40.19999 ], [ -91.49491, 40.20003 ], [ -91.50535, 40.20049 ], [ -91.50549, 40.19561 ], [ -91.51308, 40.17854 ], [ -91.50822, 40.15766 ], [ -91.51175, 40.14709 ], [ -91.5115, 40.14081 ], [ -91.51158, 40.13805 ], [ -91.51071, 40.1344 ], [ -91.51039, 40.13111 ], [ -91.5104, 40.13046 ], [ -91.51006, 40.12378 ], [ -91.50785, 40.11719 ], [ -91.50548, 40.10651 ], [ -91.50344, 40.10051 ], [ -91.50117, 40.09421 ], [ -91.49766, 40.07826 ], [ -91.48998, 40.05837 ], [ -91.49389, 40.04132 ], [ -91.48813, 40.02458 ], [ -91.48528, 40.02101 ], [ -91.47606, 40.00694 ], [ -91.47017, 39.9967 ], [ -91.46668, 39.98725 ], [ -91.46139, 39.98093 ], [ -91.45465, 39.97131 ], [ -91.44724, 39.9595 ], [ -91.43709, 39.94642 ], [ -91.43552, 39.94522 ], [ -91.43297, 39.94342 ], [ -91.42905, 39.94074 ], [ -91.42464, 39.93642 ], [ -91.42411, 39.93575 ], [ -91.42192, 39.93308 ] ] ])
    assert(county_polygons[11] == [ [ [ -89.86014, 41.54037 ], [ -89.85996, 41.53562 ], [ -89.85953, 41.5265 ],
                                      [ -89.85907, 41.51842 ], [ -89.85741, 41.5181 ], [ -89.85746, 41.51148 ], [ -89.85703, 41.47446 ], [ -89.85682, 41.4522 ], [ -89.8569, 41.40909 ], [ -89.85685, 41.38668 ], [ -89.85681, 41.36321 ], [ -89.85673, 41.35034 ], [ -89.85679, 41.33962 ], [ -89.85676, 41.32854 ], [ -89.85662, 41.32193 ], [ -89.85662, 41.31382 ], [ -89.85669, 41.30907 ], [ -89.85679, 41.30016 ], [ -89.85691, 41.28523 ], [ -89.85721, 41.27327 ], [ -89.85739, 41.2646 ], [ -89.85756, 41.25474 ], [ -89.85751, 41.24408 ], [ -89.85758, 41.2347 ], [ -89.8578, 41.23448 ], [ -89.85324, 41.23438 ], [ -89.84862, 41.23435 ], [ -89.8107, 41.23422 ], [ -89.77169, 41.23416 ], [ -89.76185, 41.23401 ], [ -89.75288, 41.23387 ], [ -89.7414, 41.23393 ], [ -89.73027, 41.23394 ], [ -89.70823, 41.23389 ], [ -89.69485, 41.23384 ], [ -89.68794, 41.23385 ], [ -89.66985, 41.23385 ], [ -89.66838, 41.23385 ], [ -89.66675, 41.23385 ], [ -89.6666, 41.23385 ], [ -89.66414, 41.23385 ], [ -89.65946, 41.23385 ], [ -89.65894, 41.23385 ], [ -89.64871, 41.23384 ], [ -89.64389, 41.23386 ], [ -89.63886, 41.23385 ], [ -89.63868, 41.21147 ], [ -89.63864, 41.19207 ], [ -89.63864, 41.17764 ], [ -89.63862, 41.16051 ], [ -89.63852, 41.14863 ], [ -89.63843, 41.14859 ], [ -89.63777, 41.14855 ], [ -89.63339, 41.14858 ], [ -89.63072, 41.14859 ], [ -89.62992, 41.1486 ], [ -89.6261, 41.14861 ], [ -89.62356, 41.14862 ], [ -89.6227, 41.14863 ], [ -89.62006, 41.14863 ], [ -89.60836, 41.14866 ], [ -89.60283, 41.14867 ], [ -89.60174, 41.14867 ], [ -89.5968, 41.14868 ], [ -89.58199, 41.1487 ], [ -89.56691, 41.14869 ], [ -89.56224, 41.14868 ], [ -89.54683, 41.14863 ], [ -89.53964, 41.14863 ], [ -89.53001, 41.14865 ], [ -89.52278, 41.14866 ], [ -89.51455, 41.14864 ], [ -89.50988, 41.14862 ], [ -89.5037, 41.14858 ], [ -89.49419, 41.14857 ], [ -89.48906, 41.14854 ], [ -89.47954, 41.14857 ], [ -89.46809, 41.14856 ], [ -89.46642, 41.14856 ], [ -89.46643, 41.15483 ], [ -89.46647, 41.19006 ], [ -89.4665, 41.20941 ], [ -89.45904, 41.23386 ], [ -89.42404, 41.23376 ], [ -89.39395, 41.2337 ], [ -89.35062, 41.25099 ], [ -89.34891, 41.25742 ], [ -89.34665, 41.26228 ], [ -89.34315, 41.2685 ], [ -89.33854, 41.27728 ], [ -89.33771, 41.28467 ], [ -89.34218, 41.29202 ], [ -89.33802, 41.2997 ], [ -89.32902, 41.30178 ], [ -89.31155, 41.30563 ], [ -89.29742, 41.3095 ], [ -89.28598, 41.31173 ], [ -89.2797, 41.31468 ], [ -89.27392, 41.31926 ], [ -89.2662, 41.32225 ], [ -89.25368, 41.32124 ], [ -89.23648, 41.31694 ], [ -89.20864, 41.31217 ], [ -89.1933, 41.3106 ], [ -89.16528, 41.30979 ], [ -89.1637, 41.31019 ], [ -89.16371, 41.31027 ], [ -89.164, 41.32437 ], [ -89.16414, 41.33562 ], [ -89.16431, 41.34751 ], [ -89.16439, 41.35386 ], [ -89.16449, 41.35833 ], [ -89.16462, 41.36436 ], [ -89.16478, 41.37367 ], [ -89.16492, 41.38366 ], [ -89.16513, 41.39507 ], [ -89.16529, 41.40818 ], [ -89.16554, 41.42132 ], [ -89.16575, 41.43016 ], [ -89.16605, 41.45242 ], [ -89.16617, 41.46351 ], [ -89.16638, 41.4804 ], [ -89.16649, 41.49782 ], [ -89.16654, 41.51063 ], [ -89.16658, 41.52491 ], [ -89.16658, 41.53554 ], [ -89.16651, 41.54806 ], [ -89.16648, 41.55362 ], [ -89.16652, 41.56509 ], [ -89.16653, 41.57221 ], [ -89.16656, 41.58442 ], [ -89.16656, 41.58528 ], [ -89.16656, 41.58529 ], [ -89.16677, 41.58529 ], [ -89.18291, 41.58533 ], [ -89.19875, 41.5852 ], [ -89.22777, 41.58536 ], [ -89.23148, 41.58521 ], [ -89.27692, 41.58512 ], [ -89.30671, 41.58499 ], [ -89.32684, 41.58498 ], [ -89.34632, 41.58494 ], [ -89.36008, 41.58493 ], [ -89.36746, 41.58498 ], [ -89.38449, 41.58495 ], [ -89.39862, 41.58493 ], [ -89.41313, 41.58496 ], [ -89.43024, 41.58499 ], [ -89.44639, 41.58492 ], [ -89.45724, 41.58496 ], [ -89.46838, 41.5849 ], [ -89.4872, 41.58496 ], [ -89.50161, 41.58501 ], [ -89.52008, 41.58503 ], [ -89.54984, 41.58508 ], [ -89.56604, 41.58506 ], [ -89.59117, 41.58506 ], [ -89.61362, 41.58505 ], [ -89.63147, 41.58495 ], [ -89.63149, 41.58495 ], [ -89.63182, 41.58494 ], [ -89.63867, 41.58495 ], [ -89.64525, 41.58494 ], [ -89.65281, 41.58493 ], [ -89.66067, 41.5849 ], [ -89.66819, 41.58486 ], [ -89.67875, 41.58479 ], [ -89.681, 41.58476 ], [ -89.68838, 41.58471 ], [ -89.70357, 41.58457 ], [ -89.70843, 41.58462 ], [ -89.72362, 41.58456 ], [ -89.73866, 41.58454 ], [ -89.74709, 41.58447 ], [ -89.75592, 41.58442 ], [ -89.76386, 41.58434 ], [ -89.76644, 41.58432 ], [ -89.78091, 41.58424 ], [ -89.78915, 41.58421 ], [ -89.80404, 41.58414 ], [ -89.80899, 41.58413 ], [ -89.82232, 41.5841 ], [ -89.82858, 41.58407 ], [ -89.83956, 41.58407 ], [ -89.85062, 41.58406 ], [ -89.86235, 41.58401 ], [ -89.86235, 41.584 ], [ -89.86084, 41.55459 ], [ -89.86015, 41.54038 ], [ -89.86014, 41.54037 ] ] ] )
    assert(county_polygons[37] == [ [ [ -88.94187, 42.01911 ], [ -88.94229, 42.00003 ], [ -88.9419, 41.98036 ],
                                      [ -88.9419, 41.97826 ], [ -88.94182, 41.96797 ], [ -88.94177, 41.95845 ], [ -88.94174, 41.95041 ], [ -88.94169, 41.94406 ], [ -88.94166, 41.93961 ], [ -88.94168, 41.93182 ], [ -88.94153, 41.92077 ], [ -88.94145, 41.9063 ], [ -88.94128, 41.89175 ], [ -88.94129, 41.88603 ], [ -88.94128, 41.87501 ], [ -88.9413, 41.86292 ], [ -88.94133, 41.85294 ], [ -88.94135, 41.8447 ], [ -88.94136, 41.8336 ], [ -88.94136, 41.82384 ], [ -88.94136, 41.80495 ], [ -88.94134, 41.79774 ], [ -88.94134, 41.79711 ], [ -88.94135, 41.79511 ], [ -88.94135, 41.7919 ], [ -88.94136, 41.7899 ], [ -88.94142, 41.78613 ], [ -88.9415, 41.76153 ], [ -88.94155, 41.75003 ], [ -88.94163, 41.72759 ], [ -88.94165, 41.71979 ], [ -88.94151, 41.71726 ], [ -88.94056, 41.71715 ], [ -88.94026, 41.71381 ], [ -88.93967, 41.68805 ], [ -88.93953, 41.66911 ], [ -88.93923, 41.65708 ], [ -88.93895, 41.64239 ], [ -88.93865, 41.62975 ], [ -88.93862, 41.62832 ], [ -88.93703, 41.62836 ], [ -88.89032, 41.6296 ], [ -88.87166, 41.63003 ], [ -88.85265, 41.63059 ], [ -88.82019, 41.63133 ], [ -88.81696, 41.63135 ], [ -88.79362, 41.63117 ], [ -88.77039, 41.63098 ], [ -88.75359, 41.63084 ], [ -88.73172, 41.63066 ], [ -88.71719, 41.63053 ], [ -88.70647, 41.63042 ], [ -88.70086, 41.63064 ], [ -88.69259, 41.6307 ], [ -88.68747, 41.63075 ], [ -88.68351, 41.6308 ], [ -88.67987, 41.63081 ], [ -88.6751, 41.63087 ], [ -88.6674, 41.63097 ], [ -88.66355, 41.631 ], [ -88.66109, 41.63099 ], [ -88.65419, 41.63104 ], [ -88.65071, 41.63099 ], [ -88.64097, 41.63111 ], [ -88.62291, 41.63127 ], [ -88.61185, 41.63133 ], [ -88.60224, 41.63139 ], [ -88.60241, 41.63867 ], [ -88.6024, 41.6387 ], [ -88.60236, 41.64227 ], [ -88.60231, 41.64238 ], [ -88.60231, 41.64245 ], [ -88.6023, 41.6428 ], [ -88.60231, 41.64461 ], [ -88.60232, 41.64592 ], [ -88.60234, 41.64719 ], [ -88.60234, 41.64759 ], [ -88.60236, 41.64853 ], [ -88.60243, 41.65321 ], [ -88.60246, 41.65491 ], [ -88.60252, 41.65903 ], [ -88.60253, 41.65986 ], [ -88.60259, 41.66572 ], [ -88.6026, 41.66756 ], [ -88.60277, 41.67438 ], [ -88.60285, 41.68184 ], [ -88.60286, 41.68223 ], [ -88.60293, 41.68654 ], [ -88.60293, 41.68663 ], [ -88.60296, 41.68837 ], [ -88.60313, 41.6968 ], [ -88.60327, 41.70407 ], [ -88.60362, 41.71955 ], [ -88.60193, 41.71956 ], [ -88.60195, 41.7235 ], [ -88.60199, 41.73427 ], [ -88.60214, 41.76913 ], [ -88.60227, 41.79142 ], [ -88.60227, 41.81399 ], [ -88.60213, 41.83387 ], [ -88.6019, 41.85039 ], [ -88.60183, 41.86066 ], [ -88.60162, 41.87936 ], [ -88.60143, 41.89636 ], [ -88.60139, 41.90466 ], [ -88.60134, 41.90848 ], [ -88.60135, 41.91462 ], [ -88.60139, 41.93373 ], [ -88.60141, 41.95863 ], [ -88.60153, 42.01052 ], [ -88.60175, 42.04418 ], [ -88.60196, 42.06647 ], [ -88.58837, 42.08823 ], [ -88.58834, 42.12003 ], [ -88.5884, 42.12226 ], [ -88.58842, 42.12424 ], [ -88.58835, 42.12579 ], [ -88.58836, 42.12706 ], [ -88.58843, 42.12852 ], [ -88.58866, 42.15357 ], [ -88.58866, 42.15359 ], [ -88.59284, 42.15356 ], [ -88.59328, 42.15355 ], [ -88.61198, 42.15347 ], [ -88.61203, 42.15347 ], [ -88.61218, 42.15345 ], [ -88.61766, 42.15341 ], [ -88.6177, 42.15342 ], [ -88.62225, 42.15345 ], [ -88.62399, 42.15347 ], [ -88.62515, 42.15347 ], [ -88.63051, 42.15346 ], [ -88.63112, 42.15346 ], [ -88.63356, 42.15346 ], [ -88.63518, 42.15346 ], [ -88.63641, 42.15346 ], [ -88.63792, 42.15346 ], [ -88.64233, 42.15346 ], [ -88.6464, 42.15346 ], [ -88.64647, 42.15345 ], [ -88.647, 42.15345 ], [ -88.64719, 42.15345 ], [ -88.65268, 42.15346 ], [ -88.66639, 42.15344 ], [ -88.66647, 42.15344 ], [ -88.67472, 42.15346 ], [ -88.69568, 42.15352 ], [ -88.70563, 42.15356 ], [ -88.71637, 42.15357 ], [ -88.73984, 42.15358 ], [ -88.74995, 42.15358 ], [ -88.76873, 42.15346 ], [ -88.77646, 42.15359 ], [ -88.77665, 42.15352 ], [ -88.77674, 42.15351 ], [ -88.78373, 42.15351 ], [ -88.79387, 42.15351 ], [ -88.80257, 42.15347 ], [ -88.80322, 42.15348 ], [ -88.82277, 42.15333 ], [ -88.82832, 42.15332 ], [ -88.83145, 42.15331 ], [ -88.83232, 42.15329 ], [ -88.83636, 42.15321 ], [ -88.8374, 42.15323 ], [ -88.8381, 42.15321 ], [ -88.84794, 42.1531 ], [ -88.85788, 42.15294 ], [ -88.86098, 42.15289 ], [ -88.86807, 42.15285 ], [ -88.87622, 42.15298 ], [ -88.88778, 42.15291 ], [ -88.9022, 42.15267 ], [ -88.93495, 42.15239 ], [ -88.93973, 42.15232 ], [ -88.93971, 42.14808 ], [ -88.93972, 42.1473 ], [ -88.93985, 42.14527 ], [ -88.93969, 42.14105 ], [ -88.93953, 42.12322 ], [ -88.93944, 42.11375 ], [ -88.93929, 42.09785 ], [ -88.93916, 42.08692 ], [ -88.93909, 42.07974 ], [ -88.94211, 42.06505 ], [ -88.9421, 42.05837 ], [ -88.94204, 42.04441 ], [ -88.94189, 42.02462 ], [ -88.94192, 42.0199 ], [ -88.94187, 42.01911 ] ] ] )

    # Print completion statement
    print("Test for county polygons generation is complete")

"""
Test function: define_county_geometry
@Params: None
@Return: None
"""
def test_define_county_geometry():

    # generate county_polygons
    county_polygons = read_county_GeoJSON("json_store/Illinois_counties.geojson")

    # sample coordinate from reference
    sample_coordinate = [
            [
                [-121.59290313720705, 37.93444993515032],
                [-121.27017974853516, 37.93444993515032],
                [-121.27017974853516, 38.065932950547484],
                [-121.59290313720705, 38.065932950547484],
                [-121.59290313720705, 37.93444993515032]
            ]
        ]

    # test polygon shape
    for FIPS, coordinates in county_polygons.items():
        assert (depth(county_polygons[FIPS]) == depth(sample_coordinate))

    # sample test cases
    FIP_1  = [ [ [ -91.42192, 39.93308 ], [ -91.42073, 39.93123 ], [ -91.41907, 39.92574 ],
                                    [ -91.41889, 39.92141 ], [ -91.41988, 39.91653 ], [ -91.4218, 39.91398 ], [ -91.42371, 39.91214 ], [ -91.43441, 39.90242 ], [ -91.44351, 39.89358 ], [ -91.44711, 39.882 ], [ -91.44638, 39.87039 ], [ -91.43733, 39.84946 ], [ -91.43653, 39.84708 ], [ -91.42149, 39.83372 ], [ -91.40509, 39.82554 ], [ -91.39314, 39.81913 ], [ -91.37515, 39.80886 ], [ -91.36848, 39.80102 ], [ -91.3618, 39.78873 ], [ -91.36347, 39.78242 ], [ -91.36569, 39.77491 ], [ -91.36591, 39.76496 ], [ -91.36512, 39.75872 ], [ -91.35264, 39.75858 ], [ -91.33798, 39.75856 ], [ -91.32164, 39.75754 ], [ -91.30514, 39.75799 ], [ -91.2646, 39.75723 ], [ -91.21991, 39.75702 ], [ -91.19039, 39.75693 ], [ -91.14878, 39.75705 ], [ -91.14806, 39.75706 ], [ -91.09738, 39.7572 ], [ -91.08186, 39.75715 ], [ -91.0581, 39.75715 ], [ -91.041, 39.75712 ], [ -91.0291, 39.75699 ], [ -91.01508, 39.757 ], [ -91.0092, 39.75701 ], [ -90.99344, 39.75687 ], [ -90.97512, 39.75687 ], [ -90.96369, 39.75698 ], [ -90.95522, 39.75707 ], [ -90.94311, 39.75713 ], [ -90.92963, 39.7572 ], [ -90.91638, 39.78915 ], [ -90.91643, 39.79652 ], [ -90.91626, 39.80433 ], [ -90.9165, 39.83038 ], [ -90.91661, 39.84507 ], [ -90.91666, 39.8522 ], [ -90.91674, 39.85621 ], [ -90.91686, 39.86913 ], [ -90.91677, 39.882 ], [ -90.91684, 39.88773 ], [ -90.91677, 39.89519 ], [ -90.9168, 39.90362 ], [ -90.91663, 39.91146 ], [ -90.91607, 39.91765 ], [ -90.91453, 39.92919 ], [ -90.91427, 39.9383 ], [ -90.91453, 39.95163 ], [ -90.91447, 39.97384 ], [ -90.91445, 39.98856 ], [ -90.91443, 40.00005 ], [ -90.91445, 40.0247 ], [ -90.91444, 40.02668 ], [ -90.91425, 40.03541 ], [ -90.91407, 40.04509 ], [ -90.91405, 40.0602 ], [ -90.9136, 40.08396 ], [ -90.91359, 40.08771 ], [ -90.91355, 40.08845 ], [ -90.9136, 40.09026 ], [ -90.91356, 40.09361 ], [ -90.9135, 40.09793 ], [ -90.91362, 40.10445 ], [ -90.91358, 40.10713 ], [ -90.91325, 40.11946 ], [ -90.9132, 40.12219 ], [ -90.91315, 40.125 ], [ -90.9131, 40.12745 ], [ -90.91297, 40.13314 ], [ -90.91296, 40.13542 ], [ -90.91291, 40.14008 ], [ -90.9129, 40.14097 ], [ -90.91289, 40.14207 ], [ -90.91289, 40.14339 ], [ -90.91286, 40.14552 ], [ -90.9128, 40.14825 ], [ -90.91277, 40.1489 ], [ -90.9126, 40.15798 ], [ -90.91258, 40.1603 ], [ -90.91254, 40.16297 ], [ -90.91248, 40.16603 ], [ -90.91246, 40.16888 ], [ -90.91238, 40.17387 ], [ -90.91236, 40.17544 ], [ -90.91226, 40.17992 ], [ -90.91213, 40.18404 ], [ -90.91211, 40.18599 ], [ -90.91207, 40.1875 ], [ -90.912, 40.19212 ], [ -90.91197, 40.19309 ], [ -90.91556, 40.19313 ], [ -90.96548, 40.19361 ], [ -91.0003, 40.19412 ], [ -91.02742, 40.19447 ], [ -91.06609, 40.1949 ], [ -91.0956, 40.19516 ], [ -91.12627, 40.19543 ], [ -91.1453, 40.19554 ], [ -91.16657, 40.19597 ], [ -91.18173, 40.19621 ], [ -91.20241, 40.19669 ], [ -91.21166, 40.19682 ], [ -91.22439, 40.19696 ], [ -91.24309, 40.19712 ], [ -91.25883, 40.1973 ], [ -91.29456, 40.19797 ], [ -91.31376, 40.19814 ], [ -91.34057, 40.19857 ], [ -91.35341, 40.19876 ], [ -91.37749, 40.19915 ], [ -91.4035, 40.19953 ], [ -91.42512, 40.1997 ], [ -91.45392, 40.19981 ], [ -91.45512, 40.19982 ], [ -91.48091, 40.19999 ], [ -91.49491, 40.20003 ], [ -91.50535, 40.20049 ], [ -91.50549, 40.19561 ], [ -91.51308, 40.17854 ], [ -91.50822, 40.15766 ], [ -91.51175, 40.14709 ], [ -91.5115, 40.14081 ], [ -91.51158, 40.13805 ], [ -91.51071, 40.1344 ], [ -91.51039, 40.13111 ], [ -91.5104, 40.13046 ], [ -91.51006, 40.12378 ], [ -91.50785, 40.11719 ], [ -91.50548, 40.10651 ], [ -91.50344, 40.10051 ], [ -91.50117, 40.09421 ], [ -91.49766, 40.07826 ], [ -91.48998, 40.05837 ], [ -91.49389, 40.04132 ], [ -91.48813, 40.02458 ], [ -91.48528, 40.02101 ], [ -91.47606, 40.00694 ], [ -91.47017, 39.9967 ], [ -91.46668, 39.98725 ], [ -91.46139, 39.98093 ], [ -91.45465, 39.97131 ], [ -91.44724, 39.9595 ], [ -91.43709, 39.94642 ], [ -91.43552, 39.94522 ], [ -91.43297, 39.94342 ], [ -91.42905, 39.94074 ], [ -91.42464, 39.93642 ], [ -91.42411, 39.93575 ], [ -91.42192, 39.93308 ] ] ]
    FIP_11 = [ [ [ -89.86014, 41.54037 ], [ -89.85996, 41.53562 ], [ -89.85953, 41.5265 ],
                                      [ -89.85907, 41.51842 ], [ -89.85741, 41.5181 ], [ -89.85746, 41.51148 ], [ -89.85703, 41.47446 ], [ -89.85682, 41.4522 ], [ -89.8569, 41.40909 ], [ -89.85685, 41.38668 ], [ -89.85681, 41.36321 ], [ -89.85673, 41.35034 ], [ -89.85679, 41.33962 ], [ -89.85676, 41.32854 ], [ -89.85662, 41.32193 ], [ -89.85662, 41.31382 ], [ -89.85669, 41.30907 ], [ -89.85679, 41.30016 ], [ -89.85691, 41.28523 ], [ -89.85721, 41.27327 ], [ -89.85739, 41.2646 ], [ -89.85756, 41.25474 ], [ -89.85751, 41.24408 ], [ -89.85758, 41.2347 ], [ -89.8578, 41.23448 ], [ -89.85324, 41.23438 ], [ -89.84862, 41.23435 ], [ -89.8107, 41.23422 ], [ -89.77169, 41.23416 ], [ -89.76185, 41.23401 ], [ -89.75288, 41.23387 ], [ -89.7414, 41.23393 ], [ -89.73027, 41.23394 ], [ -89.70823, 41.23389 ], [ -89.69485, 41.23384 ], [ -89.68794, 41.23385 ], [ -89.66985, 41.23385 ], [ -89.66838, 41.23385 ], [ -89.66675, 41.23385 ], [ -89.6666, 41.23385 ], [ -89.66414, 41.23385 ], [ -89.65946, 41.23385 ], [ -89.65894, 41.23385 ], [ -89.64871, 41.23384 ], [ -89.64389, 41.23386 ], [ -89.63886, 41.23385 ], [ -89.63868, 41.21147 ], [ -89.63864, 41.19207 ], [ -89.63864, 41.17764 ], [ -89.63862, 41.16051 ], [ -89.63852, 41.14863 ], [ -89.63843, 41.14859 ], [ -89.63777, 41.14855 ], [ -89.63339, 41.14858 ], [ -89.63072, 41.14859 ], [ -89.62992, 41.1486 ], [ -89.6261, 41.14861 ], [ -89.62356, 41.14862 ], [ -89.6227, 41.14863 ], [ -89.62006, 41.14863 ], [ -89.60836, 41.14866 ], [ -89.60283, 41.14867 ], [ -89.60174, 41.14867 ], [ -89.5968, 41.14868 ], [ -89.58199, 41.1487 ], [ -89.56691, 41.14869 ], [ -89.56224, 41.14868 ], [ -89.54683, 41.14863 ], [ -89.53964, 41.14863 ], [ -89.53001, 41.14865 ], [ -89.52278, 41.14866 ], [ -89.51455, 41.14864 ], [ -89.50988, 41.14862 ], [ -89.5037, 41.14858 ], [ -89.49419, 41.14857 ], [ -89.48906, 41.14854 ], [ -89.47954, 41.14857 ], [ -89.46809, 41.14856 ], [ -89.46642, 41.14856 ], [ -89.46643, 41.15483 ], [ -89.46647, 41.19006 ], [ -89.4665, 41.20941 ], [ -89.45904, 41.23386 ], [ -89.42404, 41.23376 ], [ -89.39395, 41.2337 ], [ -89.35062, 41.25099 ], [ -89.34891, 41.25742 ], [ -89.34665, 41.26228 ], [ -89.34315, 41.2685 ], [ -89.33854, 41.27728 ], [ -89.33771, 41.28467 ], [ -89.34218, 41.29202 ], [ -89.33802, 41.2997 ], [ -89.32902, 41.30178 ], [ -89.31155, 41.30563 ], [ -89.29742, 41.3095 ], [ -89.28598, 41.31173 ], [ -89.2797, 41.31468 ], [ -89.27392, 41.31926 ], [ -89.2662, 41.32225 ], [ -89.25368, 41.32124 ], [ -89.23648, 41.31694 ], [ -89.20864, 41.31217 ], [ -89.1933, 41.3106 ], [ -89.16528, 41.30979 ], [ -89.1637, 41.31019 ], [ -89.16371, 41.31027 ], [ -89.164, 41.32437 ], [ -89.16414, 41.33562 ], [ -89.16431, 41.34751 ], [ -89.16439, 41.35386 ], [ -89.16449, 41.35833 ], [ -89.16462, 41.36436 ], [ -89.16478, 41.37367 ], [ -89.16492, 41.38366 ], [ -89.16513, 41.39507 ], [ -89.16529, 41.40818 ], [ -89.16554, 41.42132 ], [ -89.16575, 41.43016 ], [ -89.16605, 41.45242 ], [ -89.16617, 41.46351 ], [ -89.16638, 41.4804 ], [ -89.16649, 41.49782 ], [ -89.16654, 41.51063 ], [ -89.16658, 41.52491 ], [ -89.16658, 41.53554 ], [ -89.16651, 41.54806 ], [ -89.16648, 41.55362 ], [ -89.16652, 41.56509 ], [ -89.16653, 41.57221 ], [ -89.16656, 41.58442 ], [ -89.16656, 41.58528 ], [ -89.16656, 41.58529 ], [ -89.16677, 41.58529 ], [ -89.18291, 41.58533 ], [ -89.19875, 41.5852 ], [ -89.22777, 41.58536 ], [ -89.23148, 41.58521 ], [ -89.27692, 41.58512 ], [ -89.30671, 41.58499 ], [ -89.32684, 41.58498 ], [ -89.34632, 41.58494 ], [ -89.36008, 41.58493 ], [ -89.36746, 41.58498 ], [ -89.38449, 41.58495 ], [ -89.39862, 41.58493 ], [ -89.41313, 41.58496 ], [ -89.43024, 41.58499 ], [ -89.44639, 41.58492 ], [ -89.45724, 41.58496 ], [ -89.46838, 41.5849 ], [ -89.4872, 41.58496 ], [ -89.50161, 41.58501 ], [ -89.52008, 41.58503 ], [ -89.54984, 41.58508 ], [ -89.56604, 41.58506 ], [ -89.59117, 41.58506 ], [ -89.61362, 41.58505 ], [ -89.63147, 41.58495 ], [ -89.63149, 41.58495 ], [ -89.63182, 41.58494 ], [ -89.63867, 41.58495 ], [ -89.64525, 41.58494 ], [ -89.65281, 41.58493 ], [ -89.66067, 41.5849 ], [ -89.66819, 41.58486 ], [ -89.67875, 41.58479 ], [ -89.681, 41.58476 ], [ -89.68838, 41.58471 ], [ -89.70357, 41.58457 ], [ -89.70843, 41.58462 ], [ -89.72362, 41.58456 ], [ -89.73866, 41.58454 ], [ -89.74709, 41.58447 ], [ -89.75592, 41.58442 ], [ -89.76386, 41.58434 ], [ -89.76644, 41.58432 ], [ -89.78091, 41.58424 ], [ -89.78915, 41.58421 ], [ -89.80404, 41.58414 ], [ -89.80899, 41.58413 ], [ -89.82232, 41.5841 ], [ -89.82858, 41.58407 ], [ -89.83956, 41.58407 ], [ -89.85062, 41.58406 ], [ -89.86235, 41.58401 ], [ -89.86235, 41.584 ], [ -89.86084, 41.55459 ], [ -89.86015, 41.54038 ], [ -89.86014, 41.54037 ] ] ]
    FIP_37 = [ [ [ -88.94187, 42.01911 ], [ -88.94229, 42.00003 ], [ -88.9419, 41.98036 ],
                                      [ -88.9419, 41.97826 ], [ -88.94182, 41.96797 ], [ -88.94177, 41.95845 ], [ -88.94174, 41.95041 ], [ -88.94169, 41.94406 ], [ -88.94166, 41.93961 ], [ -88.94168, 41.93182 ], [ -88.94153, 41.92077 ], [ -88.94145, 41.9063 ], [ -88.94128, 41.89175 ], [ -88.94129, 41.88603 ], [ -88.94128, 41.87501 ], [ -88.9413, 41.86292 ], [ -88.94133, 41.85294 ], [ -88.94135, 41.8447 ], [ -88.94136, 41.8336 ], [ -88.94136, 41.82384 ], [ -88.94136, 41.80495 ], [ -88.94134, 41.79774 ], [ -88.94134, 41.79711 ], [ -88.94135, 41.79511 ], [ -88.94135, 41.7919 ], [ -88.94136, 41.7899 ], [ -88.94142, 41.78613 ], [ -88.9415, 41.76153 ], [ -88.94155, 41.75003 ], [ -88.94163, 41.72759 ], [ -88.94165, 41.71979 ], [ -88.94151, 41.71726 ], [ -88.94056, 41.71715 ], [ -88.94026, 41.71381 ], [ -88.93967, 41.68805 ], [ -88.93953, 41.66911 ], [ -88.93923, 41.65708 ], [ -88.93895, 41.64239 ], [ -88.93865, 41.62975 ], [ -88.93862, 41.62832 ], [ -88.93703, 41.62836 ], [ -88.89032, 41.6296 ], [ -88.87166, 41.63003 ], [ -88.85265, 41.63059 ], [ -88.82019, 41.63133 ], [ -88.81696, 41.63135 ], [ -88.79362, 41.63117 ], [ -88.77039, 41.63098 ], [ -88.75359, 41.63084 ], [ -88.73172, 41.63066 ], [ -88.71719, 41.63053 ], [ -88.70647, 41.63042 ], [ -88.70086, 41.63064 ], [ -88.69259, 41.6307 ], [ -88.68747, 41.63075 ], [ -88.68351, 41.6308 ], [ -88.67987, 41.63081 ], [ -88.6751, 41.63087 ], [ -88.6674, 41.63097 ], [ -88.66355, 41.631 ], [ -88.66109, 41.63099 ], [ -88.65419, 41.63104 ], [ -88.65071, 41.63099 ], [ -88.64097, 41.63111 ], [ -88.62291, 41.63127 ], [ -88.61185, 41.63133 ], [ -88.60224, 41.63139 ], [ -88.60241, 41.63867 ], [ -88.6024, 41.6387 ], [ -88.60236, 41.64227 ], [ -88.60231, 41.64238 ], [ -88.60231, 41.64245 ], [ -88.6023, 41.6428 ], [ -88.60231, 41.64461 ], [ -88.60232, 41.64592 ], [ -88.60234, 41.64719 ], [ -88.60234, 41.64759 ], [ -88.60236, 41.64853 ], [ -88.60243, 41.65321 ], [ -88.60246, 41.65491 ], [ -88.60252, 41.65903 ], [ -88.60253, 41.65986 ], [ -88.60259, 41.66572 ], [ -88.6026, 41.66756 ], [ -88.60277, 41.67438 ], [ -88.60285, 41.68184 ], [ -88.60286, 41.68223 ], [ -88.60293, 41.68654 ], [ -88.60293, 41.68663 ], [ -88.60296, 41.68837 ], [ -88.60313, 41.6968 ], [ -88.60327, 41.70407 ], [ -88.60362, 41.71955 ], [ -88.60193, 41.71956 ], [ -88.60195, 41.7235 ], [ -88.60199, 41.73427 ], [ -88.60214, 41.76913 ], [ -88.60227, 41.79142 ], [ -88.60227, 41.81399 ], [ -88.60213, 41.83387 ], [ -88.6019, 41.85039 ], [ -88.60183, 41.86066 ], [ -88.60162, 41.87936 ], [ -88.60143, 41.89636 ], [ -88.60139, 41.90466 ], [ -88.60134, 41.90848 ], [ -88.60135, 41.91462 ], [ -88.60139, 41.93373 ], [ -88.60141, 41.95863 ], [ -88.60153, 42.01052 ], [ -88.60175, 42.04418 ], [ -88.60196, 42.06647 ], [ -88.58837, 42.08823 ], [ -88.58834, 42.12003 ], [ -88.5884, 42.12226 ], [ -88.58842, 42.12424 ], [ -88.58835, 42.12579 ], [ -88.58836, 42.12706 ], [ -88.58843, 42.12852 ], [ -88.58866, 42.15357 ], [ -88.58866, 42.15359 ], [ -88.59284, 42.15356 ], [ -88.59328, 42.15355 ], [ -88.61198, 42.15347 ], [ -88.61203, 42.15347 ], [ -88.61218, 42.15345 ], [ -88.61766, 42.15341 ], [ -88.6177, 42.15342 ], [ -88.62225, 42.15345 ], [ -88.62399, 42.15347 ], [ -88.62515, 42.15347 ], [ -88.63051, 42.15346 ], [ -88.63112, 42.15346 ], [ -88.63356, 42.15346 ], [ -88.63518, 42.15346 ], [ -88.63641, 42.15346 ], [ -88.63792, 42.15346 ], [ -88.64233, 42.15346 ], [ -88.6464, 42.15346 ], [ -88.64647, 42.15345 ], [ -88.647, 42.15345 ], [ -88.64719, 42.15345 ], [ -88.65268, 42.15346 ], [ -88.66639, 42.15344 ], [ -88.66647, 42.15344 ], [ -88.67472, 42.15346 ], [ -88.69568, 42.15352 ], [ -88.70563, 42.15356 ], [ -88.71637, 42.15357 ], [ -88.73984, 42.15358 ], [ -88.74995, 42.15358 ], [ -88.76873, 42.15346 ], [ -88.77646, 42.15359 ], [ -88.77665, 42.15352 ], [ -88.77674, 42.15351 ], [ -88.78373, 42.15351 ], [ -88.79387, 42.15351 ], [ -88.80257, 42.15347 ], [ -88.80322, 42.15348 ], [ -88.82277, 42.15333 ], [ -88.82832, 42.15332 ], [ -88.83145, 42.15331 ], [ -88.83232, 42.15329 ], [ -88.83636, 42.15321 ], [ -88.8374, 42.15323 ], [ -88.8381, 42.15321 ], [ -88.84794, 42.1531 ], [ -88.85788, 42.15294 ], [ -88.86098, 42.15289 ], [ -88.86807, 42.15285 ], [ -88.87622, 42.15298 ], [ -88.88778, 42.15291 ], [ -88.9022, 42.15267 ], [ -88.93495, 42.15239 ], [ -88.93973, 42.15232 ], [ -88.93971, 42.14808 ], [ -88.93972, 42.1473 ], [ -88.93985, 42.14527 ], [ -88.93969, 42.14105 ], [ -88.93953, 42.12322 ], [ -88.93944, 42.11375 ], [ -88.93929, 42.09785 ], [ -88.93916, 42.08692 ], [ -88.93909, 42.07974 ], [ -88.94211, 42.06505 ], [ -88.9421, 42.05837 ], [ -88.94204, 42.04441 ], [ -88.94189, 42.02462 ], [ -88.94192, 42.0199 ], [ -88.94187, 42.01911 ] ] ]
    test_FIPS = [FIP_1, FIP_11, FIP_37]
    for FIP_coordinates in test_FIPS:
        geojson = define_county_geometry(FIP_coordinates)
        assert(depth(geojson["coordinates"]) == depth(sample_coordinate))
    print("Completed verification for define_county_geometry")

"""
Test function: test define_county_geometry picture generation
@Params: None
@Return: None
"""
def test_define_county_geometry_pic_generation():
    PLANET_API_KEY = "b99bfe8b97d54205bccad513987bbc02"
    combined_filter = test_filter()
    assetResultsVector = downAssets.extract_images(PLANET_API_KEY, 2, combined_filter)
    results = downAssets.parallelize(assetResultsVector)
    for tensor in results:
        plt.figure()
        plt.imshow(tensor[0]) # Output the R-band
        plt.show()
    print("Completed verification for picture generation from county geometry")
"""
Helper Test function: define custom filter (TODO)
@Params: n/a
@Return: filter
"""
def test_filter():
    # generate county_polygons and filter
    county_polygons = read_county_GeoJSON("json_store/Illinois_counties.geojson")
    geojson = fetchData.define_geometry()
    # geojson = define_county_geometry(county_polygons[161])

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
            "lte": "2016-10-01T00:00:00.000Z"
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
Helper Test function: check depth of nested list
@Params: sequence
@Return: Level of nesting
Reference: https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
"""
def depth(seq):
    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

"""
Test Function: verifies the binning functions
@Params: n/a
@Return: n/a
"""
def test_data_distribution():
    print("--- starting test for data distribution ---")
    bins_array = truth_data_distribution(filename = "json_store/Illinois_Soybeans_Truth_Data.csv", num_classes = 10)
    test_array = np.array([20, 38, 45])
    print(bin_truth_data(test_array, bins_array)) # should print 1, 2, 4
    print("--- test complete ---")

"""
Function: Determine distribution of truth data and split output range of classes 
@Params: county truth dict[County FIP: dict[year, yield]]
@Return: Array form of quantile split of data
"""
def truth_data_distribution(filename: str, num_classes = 10):

    # Extract all yields
    yields = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        next(reader)
        for row in reader:
            yields.append(float(row[-2]))

    # Perform the quantile cut
    np.random.seed(42)
    test_list_rnd = np.array(yields) + np.random.random(len(yields))  # add noise to data
    test_series = pd.Series(test_list_rnd, name='value_rank')
    split_df = pd.qcut(test_series, q=num_classes, retbins=True, labels=False)
    # split_df = pd.qcut(yields, q=num_classes)
    bins = split_df[1]
    return bins

"""
Function: Converts master label dict with dict(key: id, value: yield/county) to dict(key: id, value: label)
"""
def convert_master_label_dict(num_classes):

    # extract pickle file
    label_dict = {}
    with open('json_store/labels/master_label_dict.pkl', 'rb') as fp:
        label_dict = pickle.load(fp) # dictionary of {'id-1': label 1, ... , 'id-n', label n}
    values = list(label_dict.values())

    # Perform the quantile cut
    np.random.seed(42)
    test_list_rnd = np.array(values) + np.random.random(len(values))  # add noise to data
    test_series = pd.Series(test_list_rnd, name='value_rank')
    split_df = pd.qcut(test_series, q=num_classes, retbins=True, labels=False)
    # split_df = pd.qcut(yields, q=num_classes)
    bins = split_df[1]

    # bin the original dictionary
    for key, val in label_dict.items():
        [cropLabel] = bin_truth_data(np.array([val]), bins)
        label_dict[key] = cropLabel  # convert to label

    # Write to pickle file
    with open('json_store/labels/master_label_dict_binned.pkl', 'wb') as fp:
        pickle.dump(label_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

