# http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/

import requests
# import urllib.request
from bs4 import BeautifulSoup
import re
import os
import urllib.parse
import numpy as np

baseurl = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"

response = requests.get(baseurl)
soup = BeautifulSoup(response.text, 'html.parser')
individual_pages = soup.findAll('a')

detect_url = lambda x: re.findall(r'(?<=a href=".\/)distribute\/v1\.0\/elv.*(?=")', x)
urls = [detect_url(str(x)) for x in individual_pages]
urls = list(filter(None, urls))

# target = r'n30w090' # north america
# target = r's60w180' # oceania
detect_target = lambda x: re.findall(target, x)
target_position = np.argmax([len(x) for x in [detect_target(str(x)) for x in urls]])

url = baseurl + urls[target_position][0]

# elv_n30w090.tar 
# url = 'http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0/elv_s60w180.tar'
filename = os.path.basename(urllib.parse.urlparse(url).path)
username = "hydrography"
password = "rivernetwork"

# ---
# https://stackoverflow.com/a/47342052/3362993
import tqdm

def download_file(url, filename):
    
    r = requests.get(url, auth=(username,password), stream=True)
    total_size = int(r.headers.get('content-length', 0))

    with open(filename, 'wb') as f:
        for chunk in tqdm.tqdm(r.iter_content(32*1024), total=total_size,unit='B', unit_scale=True):
            if chunk:
                f.write(chunk)

    return path

download_file(url, filename)

# ---
r = requests.get(url)
r = requests.get(url, auth=(username,password), timeout = None)

if r.status_code == 200:
   with open(filename, 'wb') as out:
      for bits in r.iter_content():
          out.write(bits)

