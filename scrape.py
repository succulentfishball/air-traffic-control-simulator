import os
import requests

# URL of the OpenSky directory containing CSV links
base_url = "https://s3.opensky-network.org/data-samples/states/.2017-08-14/10/states_2017-08-14-10.csv.tar"

with requests.get(base_url, stream=True) as r:
    r.raise_for_status()
    with open("data/test2.csv.tar", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)