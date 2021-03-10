import rasterio
import boto3
from rasterio.plot import show
import matplotlib.pyplot as plt
from orders import place_order, poll_for_success, download_order
import requests
import pickle
from requests.auth import HTTPBasicAuth
import tensorflow as tf
import numpy as np
import random
import pickle5 as pickle
import time
import io
from extractCountyData import truth_data_distribution, bin_truth_data
import datetime
import os
import json
import csv

# Define global variables
AWS_SERVER_PUBLIC_KEY = "" #TODO: Add after pull
AWS_SERVER_SECRET_KEY = "" #TODO: Add after pull

# Import functions
from imageProcessing import image_process
# from orders import obtain_crop_labels
import pickle5 as pickle
import pandas as pd

#Import county extraction functions
from extractCountyData import read_county_GeoJSON, read_county_truth

"""
Integrated Function: Crawl through AWS Bucket images
1) Crawl through AWS Buckets, Identify MS_image
2) Apply IP method to MS_image, Label resulting tensor to create a tuple
3) Store into correct file location
"""
def s3ProcessLabelImage(s3_client, bucket, session, cropLabels, truth_data_distribution):
    print("STARTING S3 PROCESSING")
    labelDictionary = {}
    partition = {}
    idArray = np.array([])
    cropArray = []
    masterTensorEmptyFlag = True
    for obj in bucket.objects.all():
        fileName = obj.key
        if fileName.endswith('AnalyticMS_clip.tif'):
            print("PROCESSING REAL IMAGE")
            #Obtain crop label for image
            idSplit1 = fileName.split("Scene4Band/")[1] #After "Scene4Band"
            id = idSplit1.split("_3B_AnalyticMS_")[0] #Before "AnalyticMS"
            cropStat = cropLabels[id]
            cropArray.append(cropStat)

            #Attain image + package
            # image_tensor = np.array([image_process(session, 's3://cs230data/' + fileName)])
            start = time.time()
            image_tensor = np.array(image_process(session, 's3://cs230data/' + fileName)) # shape (C, H, W)
            image_tensor = np.transpose(image_tensor, (1, 2, 0)) # shape (H, W, C)
            end = time.time()
            print("image process complete in " + str(end - start))

            # Write image to aws
            start = time.time()
            image_tensor_data = io.BytesIO()
            pickle.dump(image_tensor, image_tensor_data)
            image_tensor_data.seek(0)
            s3_client.upload_fileobj(image_tensor_data, 'cs230data', str(id) + '.pkl')
            end = time.time()
            print("Complete upload in " + str(end - start))

            # extract labels and id
            [cropLabel] = bin_truth_data(np.array([cropStat], truth_data_distribution))
            labelDictionary[id] = cropLabel # convert to label
            idArray = np.append(idArray, id)
            print("FINISHED PROCESSING REAL IMAGE")

    #Partition: Dictionary of val, test, train keys to [Id List] containing relevant ids
    #Label Dictionary: Key is ID, Value is crop label
    shuffledIds = random.shuffle(idArray)
    ten_percent_split = (-1) * int(len(shuffledIds) / 10)
    twent_percent_split = (-2) * int(len(shuffledIds) / 10)
    test_ids = shuffledIds[ten_percent_split:]
    val_ids = shuffledIds[twent_percent_split:ten_percent_split]
    train_ids = shuffledIds[:twent_percent_split]
    partition = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    # Write data structures to file
    with open('data/partition.p', 'wb') as fp:
        pickle.dump(partition, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('labels.p', 'wb') as fp:
        pickle.dump(labelDictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return (labelDictionary, partition)

#Create partition
def createPartition(s3_client, bucket, session):
    partition = {}
    idArray = np.array([])
    for obj in bucket.objects.all():
        fileName = obj.key
        if fileName.endswith('AnalyticMS_clip.tif'):
            #Obtain crop label for image
            idSplit1 = fileName.split("Scene4Band/")[1] #After "Scene4Band"
            id = idSplit1.split("_3B_AnalyticMS_")[0] #Before "AnalyticMS"
            idArray = np.append(idArray, id)
    # Partition: Dictionary of val, test, train keys to [Id List] containing relevant ids
    # Label Dictionary: Key is ID, Value is crop label
    np.random.shuffle(idArray)
    ten_percent_split = (-1) * int(len(idArray) / 10)
    twent_percent_split = (-2) * int(len(idArray) / 10)
    test_ids = idArray[ten_percent_split:]
    val_ids = idArray[twent_percent_split:ten_percent_split]
    train_ids = idArray[:twent_percent_split]
    print("Test", len(test_ids))
    print("Train", len(train_ids))
    print("Val", len(val_ids))
    partition = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    with open('data/partition_vUpdate.p', 'wb') as fp:
        pickle.dump(partition, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return partition

#Create metadata CSV (Generate from images in the s3 buckets)
def createMetadata():
    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket('cs230datarev2')
    s3_client = session.client('s3')
    for obj in bucket.objects.all():
        fileName = obj.key
        if fileName.endswith('metadata.json'):
            content_object = s3.Object('cs230datarev2', fileName)
            file_content = content_object.get()['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)
            id = json_content['id']
            json_content = json_content['properties']

            anom_pix = json_content['anomalous_pixels']
            cloud_cover = json_content['cloud_cover']
            origin_x = json_content['origin_x']
            origin_y = json_content['origin_y']
            time = json_content['acquired']
            strip_id = json_content['strip_id']

            fieldnames = ['id', 'anomalous_pix_perc', 'cloud_cover',
                          'origin_x', 'origin_y', 'acquired', 'strip_id']
            with open(r'metadata/metadata.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(
                    {'id': id, 'anomalous_pix_perc': anom_pix, 'cloud_cover': cloud_cover,
                     'origin_x': origin_x, 'origin_y': origin_y, 'acquired': time, 'strip_id': strip_id})







#Writes master example dictionary to npy file
def writeToNpy(path, dictionary):
    np.save(path, dictionary)

#Loads npy path into dictionary of examples
def loadFromNpy(path):
    exampleDictionary = np.load(path)


"""
TODO Chris: Write Tensor to S3 folder
https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/s3.md
"""
#Tester function to write random tensor
def testerWriteRandomString(s3):
    a1 = tf.range(10)
    a2 = tf.reshape(a1, [2, 5])
    byteObject = pickle.dumps(a2)
    objectKey = "imageTensors1/" + "Hello"
    s3.put_object(Body=byteObject, Bucket='cs230data', Key=objectKey)

"""
TODO Chris: Read Tensor to S3 folder
https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/s3.md
"""
#Tester function to get image tensor
def getImageTensors(s3, key):
    objectKey = "imageTensors1/" + key
    data = s3.get_object(Bucket='cs230data', Key=objectKey)

    #Debug decoding step
    decoded = data.get('Body').decode()
    return decoded


"""
Function: This is a one-time function to generate a dictionary in the following form dict{key: id, value: aws file path}
"""
def create_file_path_directory():

    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket('cs230datarev2')

    # extract all file paths into list
    file_path_dict = {}
    for obj in bucket.objects.all():
        fileName = obj.key
        if fileName.endswith('AnalyticMS_clip.tif'):
            idSplit1 = fileName.split("Scene4Band/")[1]  # After "Scene4Band"
            id = idSplit1.split("_3B_AnalyticMS_")[0]  # Before "AnalyticMS"
            file_path_dict[id] = fileName
            print(fileName)
    # write to pickle file
    with open('data/aws_file_dict_vUpdate.p', 'wb') as fp:
        pickle.dump(file_path_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

"""
Function: Analyze all images in AWS and compute areas based on non-zero pixels
"""
def compute_image_areas():
    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket('cs230datarev2')

    # extract all file paths into list
    id_area_dict = {}
    count = 0
    for obj in bucket.objects.all():
        fileName = obj.key
        if fileName.endswith('AnalyticMS_clip.tif'):
            idSplit1 = fileName.split("Scene4Band/")[1]  # After "Scene4Band"
            id = idSplit1.split("_3B_AnalyticMS_")[0]  # Before "AnalyticMS"

            #Read in numpy array TODONOW
            with rasterio.Env(aws_secret_access_key=AWS_SERVER_SECRET_KEY,
                              aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
                              aws_session=session):
                bucket = 's3://' + 'cs230datarev2' + '/'
                path = bucket + str(fileName)
                data = rasterio.open(path)
                image = data.read()
                b, g, r, n = image
                print("created image")
                nonzero_count = np.count_nonzero(g)
                print("Counted nonzero")
                id_area_dict[id] = nonzero_count * 9 #Each pixel is 9m^2
                print("IMAGE ADDED", count)
                count += 1

    # write to pickle file
    with open('data/aws_id_areas.p', 'wb') as fp:
        pickle.dump(id_area_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

"""
Function: Extract images from s3 bucket, perform image processing, write back to the cloud
"""
def store_processed_images(maxH, maxW, scaling, completedPath='processed_images/completed_images.pkl',
                           awsFileDictPath='aws_file_dict.p', s3bucketName='cs230data', imageDirPath='processed_images/vanilla_model/'):

    # read back in saved crop list
    completed_images = []
    if (os.path.isfile(completedPath)):
        with open(completedPath, 'rb') as fp:
            completed_images = pickle.load(fp)

    # start session
    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket(s3bucketName)

    # read in pickl file for all the file paths
    with open(awsFileDictPath, 'rb') as fp:
        aws_file_dict = pickle.load(fp)
<<<<<<< HEAD
     
    count = 0 
    # loop through all the images in data set
    for key, val in aws_file_dict.items():
        count += 1

=======

    count = 0
    # loop through all the images in data set
    for key, val in aws_file_dict.items():

        count += 1
>>>>>>> c65030398ada0540e8e1091fba155b720c3a8b01
        # start timer
        start = time.time()

        # check to see if id has already been run
        if key in completed_images:
            continue # skip image if already processed

        # perform image processing
        s3BucketStr = 's3://' + s3bucketName + '/'
        image = image_process(session, s3BucketStr + str(val), maxH, maxW, scaling) # shape (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # shape (H, W, C)
        image = image / 255

        # write to local directory
        with open(imageDirPath + str(key) + '.npy', 'wb') as fp:
            pickle.dump(image, fp)

        # store to completed images to local directory
        completed_images.append(key)
        with open(completedPath, 'wb') as fp:
            pickle.dump(completed_images, fp)

        # print statements
        end = time.time()
        print("Image " + str(count) + " is complete in " + str(end - start))

if __name__ == "__main__":

    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket('cs230datarev2')
    s3_client = session.client('s3')

    compute_image_areas()

    # createMetadata()

    # create_file_path_directory()
<<<<<<< HEAD
    # createPartition(s3_client, bucket, session)
    # create_file_path_directory()
    store_processed_images(maxH=500, maxW=500, scaling=0.04, completedPath='processed_images/completed_images_v2.pkl',
                           awsFileDictPath='aws_file_dict_vUpdate2.p', s3bucketName='cs230datarev2',
                           imageDirPath='processed_images/concat_model/')
=======
    #createPartition(s3_client, bucket, session)
    #create_file_path_directory()
    #store_processed_images(maxH=500, maxW=500, scaling=0.04, completedPath='processed_images/completed_images_v2.pkl',
                           #awsFileDictPath='aws_file_dict_vUpdate.p', s3bucketName='cs230datarev2',
                           #imageDirPath='processed_images/concat_model/')
>>>>>>> c65030398ada0540e8e1091fba155b720c3a8b01











"""
Obselete Test Functions
"""


"""
Test function: This function performs a downloads a single product to the s3 bucket
@Param: none
@Return: none
"""
def test_write_to_s3():

    # test input variables
    orders_url = 'https://api.planet.com/compute/ops/orders/v2'
    PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'
    auth = HTTPBasicAuth(PLANET_API_KEY, '')

    # define product
    single_product = [
        {
            "item_ids": ["20151119_025740_0c74"],  # ***NOTE*** Code to search for item id is already done
            "item_type": "PSScene4Band",
            "product_bundle": "analytic"
        }
    ]

    # define delivery
    delivery = {
        "amazon_s3": {
            "bucket": "cs230data",
            "aws_region": "us-west-1",
            "aws_access_key_id": AWS_SERVER_PUBLIC_KEY,
            "aws_secret_access_key": AWS_SERVER_SECRET_KEY,
            "path_prefix": "test3"
        }
    }

    # define request
    request = {
        "name": "test download",
        "products": single_product,
        "delivery": delivery
    }

    # make download to s3
    order_url = place_order(request, auth)
    poll_for_success(order_url, auth)
    print("Download to AWS s3 was successful")

"""
Test function: This function attempts to open a .tif from the cloud and interact with the rgbn data and make a sample plot.
# Note: This function assumes 'test_image1.tif' is in the root of the bucket
@Param: aws session
@Return: none
"""
def test_tif(session):
    # Read in .tif test
    with rasterio.Env(aws_secret_access_key=AWS_SERVER_SECRET_KEY,
                      aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
                      aws_session=session):
        url = 's3://cs230data/test_image1.tif'
        data = rasterio.open(url)
        b, g, r, n = data.read()
        plt.imshow(b)
        plt.show()
    print("printing tif was sucessful")

    """
    Test Function: Store process images
    """

    def test_store_processed_images():
        # run store process images
        store_processed_images(maxH=500, maxW=500, scaling=0.04)

        # verify contents of test image
        with open('json_store/completed_images.pkl', 'rb') as fp:
            processed_images = pickle.load(fp)
            print("processed images")
            print(processed_images)
        with open('processed_images/vanilla_model/20191012_150709_1049.npy', 'rb') as fp:
            sample = pickle.load(fp)
            print(sample.shape)


"""
Run and store crop labels to test first county
"""
def obtainAndStoreCropLabels():
    county_dict = read_county_GeoJSON(filename='json_store/Illinois_counties.geojson')
    county_truth = read_county_truth(filename='json_store/Illinois_Soybeans_Truth_Data.csv')
    # cropLabels = obtain_crop_labels(county_dict, county_truth)

    """# temporarily comment out
    path = "json_store/labels_all"
    outfile = open(path, 'wb')
    pickle.dump(cropLabels, outfile)
    outfile.close()"""

