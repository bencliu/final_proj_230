import json
import os
import pathlib
import time

import numpy as np
import rasterio
from rasterio.plot import show
import requests
from requests.auth import HTTPBasicAuth

orders_url = 'https://api.planet.com/compute/ops/orders/v2'


# define products part of order
single_product = [
    {
      "item_ids": ["20151119_025740_0c74"],
      "item_type": "PSScene4Band",
      "product_bundle": "analytic"
    }
]