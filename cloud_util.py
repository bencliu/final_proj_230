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

# Define global variables
AWS_SERVER_PUBLIC_KEY = "AKIAIKIF2RFKG3KM3ALQ" #TODO: Add after pull
AWS_SERVER_SECRET_KEY = "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" #TODO: Add after pull

# Import functions
from imageProcessing import image_process
from orders import obtain_crop_labels

#Import county extraction functions
from extractCountyData import read_county_GeoJSON, read_county_truth

"""
Integrated Function: Crawl through AWS Bucket images
1) Crawl through AWS Buckets, Identify MS_image
2) Apply IP method to MS_image, Label resulting tensor to create a tuple
3) Store into correct file location
"""
def s3ProcessLabelImage(bucket, session, cropLabels):
    print("STARTING S3 PROCESSING")
    masterTensorArray = np.array([])
    masterLabelArray = np.array([])
    idArray = np.array([])
    for obj in bucket.objects.all():
        fileName = obj.key
        if fileName.endswith('AnalyticMS_clip.tif'):
            print("PROCESSING REAL IMAGE")
            #Obtain crop label for image
            idSplit1 = fileName.split("Scene4Band/")[1] #After "Scene4Band"
            id = idSplit1.split("_3B_AnalyticMS_")[0] #Before "AnalyticMS"
            cropStat = cropLabels[id]

            #Attain image + package
            image_tensor = np.array([image_process(session, 's3://cs230data/' + fileName)])
            print("SHAPE:", image_tensor.shape)
            assert image_tensor.shape[0] == 1
            masterTensorArray= np.append(masterTensorArray, image_tensor, axis=0)
            masterLabelArray = np.append(masterLabelArray, cropStat)
            idArray = np.append(idArray, id)
            print("FINISHED PROCESSING REAL IMAGE")
            break

    return (masterTensorArray, masterLabelArray, idArray)

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
Run and store crop labels to test first county
"""
def obtainAndStoreCropLabels():
    county_dict = read_county_GeoJSON(filename='json_store/Illinois_counties.geojson')
    county_truth = read_county_truth(filename='json_store/Illinois_Soybeans_Truth_Data.csv')
    cropLabels = obtain_crop_labels(county_dict, county_truth)

    path = "json_store/labels_c1"
    outfile = open(path, 'wb')
    pickle.dump(cropLabels, outfile)
    outfile.close()

if __name__ == "__main__":

    # Define session for aws
    session = boto3.Session(
        aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=AWS_SERVER_SECRET_KEY,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket('cs230data')

    #obtainAndStoreCropLabels()
    infile = open("json_store/labels_c1", 'rb')
    cropLabels = pickle.load(infile)
    infile.close()
    
    print("FIRST IDS", list(cropLabels.keys())[:10])
    print("FIRST CROP STATS", list(cropLabels.values())[:10])
    s3ProcessLabelImage(bucket, session, cropLabels)







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

