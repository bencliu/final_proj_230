import json
import os
import pathlib
import time

import numpy as np
import rasterio
from rasterio.plot import show
import requests
from requests.auth import HTTPBasicAuth

# global variables and setup
orders_url = 'https://api.planet.com/compute/ops/orders/v2'
PLANET_API_KEY = 'b99bfe8b97d54205bccad513987bbc02'
auth = HTTPBasicAuth(PLANET_API_KEY, '')
headers = {'content-type': 'application/json'}

# define helpful functions for submitting, polling, and downloading an order
# reference: https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/tools_and_toolchains.ipynb
def place_order(request, auth):
    response = requests.post(orders_url, data=json.dumps(request), auth=auth, headers=headers)
    print(response)

    if not response.ok:
        raise Exception(response.content)

    order_id = response.json()['id']
    print(order_id)
    order_url = orders_url + '/' + order_id
    return order_url

def poll_for_success(order_url, auth, num_loops=50):
    count = 0
    while (count < num_loops):
        count += 1
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        success_states = ['success', 'partial']
        if state == 'failed':
            raise Exception(response)
        elif state in success_states:
            break
        time.sleep(10)

def download_order(order_url, auth, overwrite=False):
    r = requests.get(order_url, auth=auth)
    print(r)
    print(r.json())
    response = r.json()
    results = response['_links']['results']
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    results_paths = [pathlib.Path(os.path.join('data', n)) for n in results_names]
    print('{} items to download'.format(len(results_urls)))

    for url, name, path in zip(results_urls, results_names, results_paths):
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name, path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping {}'.format(path, name))

    return dict(zip(results_names, results_paths))

# define helpful functions for visualizing downloaded imagery
def show_rgb(img_file):
    with rasterio.open(img_file) as src:
        b,g,r,n = src.read()
    rgb = np.stack((r,g,b), axis=0)
    show(rgb/rgb.max())

# simple download
def test_simple_download():

    # define products
    single_product = [
        {
            "item_ids": ["20151119_025740_0c74"], # ***NOTE*** Code to search for item id is already done
            "item_type": "PSScene4Band",
            "product_bundle": "analytic"
        }
    ]
    multi_product = [
        {
            "item_ids": ["20151119_025740_0c74",
                         "20151119_025739_0c74"],
            "item_type": "PSScene4Band",
            "product_bundle": "analytic"
        }
    ]

    # define clip
    clip_aoi = {
        "type": "Polygon",
        "coordinates": [[[94.81858044862747, 15.858073043526062],
                         [94.86242249608041, 15.858073043526062],
                         [94.86242249608041, 15.894323164978303],
                         [94.81858044862747, 15.894323164978303],
                         [94.81858044862747, 15.858073043526062]]]
    }
    clip = {
        "clip": {
            "aoi": clip_aoi
        }
    }

    # define delivery
    delivery = {
        "google_cloud_storage": {
            "bucket": "blah...",
            "credentials": "blah...",
            "path_prefix": "blah..."
        }
    }

    # define request
    request = {
        "name": "test download",
        "products": multi_product,
        "tools": [clip]
        # "delivery": delivery
    }

    # Perform Order
    print("starting order")
    order_url = place_order(request, auth)
    print("starting polling")
    poll_for_success(order_url, auth)
    print("starting download")
    downloaded_files = download_order(order_url, auth)
    # img_file = next(downloaded_files[d] for d in downloaded_files if d.endswith('_3B_AnalyticMS.tif'))
    img_file = next(downloaded_files[d] for d in downloaded_files if d.endswith('_clip.tif'))

    # Display image
    show_rgb(img_file)
    print("simple download test sucessful")

if __name__ == "__main__":

    # test simple download
    test_simple_download()
