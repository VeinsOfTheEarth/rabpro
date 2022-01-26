# rabpro - river and basin profiler <a href='https:///VeinsOfTheEarth.github.io/rabpro/'><img src="docs/_static/logo.png" align="right" height=140/></a>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Anaconda badge](https://anaconda.org/jschwenk/rabpro/badges/version.svg)](https://anaconda.org/jschwenk/rabpro) [![build](https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml/badge.svg)](https://github.com/VeinsOfTheEarth/rabpro/actions/workflows/build.yaml)

> Package to delineate watershed subbasins, compute statistics, and gather attributes

## Setup

### Software

```shell
conda env create -f environment.yml
source activate rabpro

# set use-feature to silence deprecation warning
# pip install --use-feature=in-tree-build . 
```

### Data

Locate the MERIT DEM "tile(s)" of interest and run the following command with username and password arguments:

```shell
rabpro download merit n30e150 <username> <password>
```

Download Hydrobasins levels 1 and 12:

```shell
rabpro download hydrobasins
```

## Usage ([documentation](https:///VeinsOfTheEarth.github.io/rabpro/))

### command line

```shell
python rabpro/cli/run_rabpro.py
```

#### python

```python
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rabpro
from rabpro.subbasin_stats import Dataset

def pull_basin(tag):
    # tag = "test_coords"
    coords_file = gpd.read_file(r"tests/data/" + tag + ".shp")
    rpo = rabpro.profiler(coords_file)
    rpo.delineate_basins()
    rpo.basins.to_file(tag + ".gpkg",driver="GPKG")    
    
    # pull gee
    url, task = rpo.basin_stats([Dataset("JRC/GSW1_3/GlobalSurfaceWater", "occurrence")])
    r = requests.get(url)
    with open(tag + "_gsw.csv", "wb") as f:
        f.write(r.content)
    
    # merge gee and gdf
    csv_gsw = pd.read_csv(tag + "_gsw.csv")
    gdf_gsw = gpd.read_file(tag + ".gpkg")
    gdf_gsw["mean"] = [float(x) for x in csv_gsw["mean"]]
    
    return gdf_gsw

res = [pull_basin(x) for x in ["test_coords", "test_coords2"]]
res = pd.concat(res)
res.plot(column = "mean", legend = True)
plt.legend(title="GSW occurrence %", loc = (0.7, 1), frameon = False)
plt.show()
```

![example output image](https://VeinsOfTheEarth.github.io/rabpro/_images/readme.png)

## Testing

```python
# file-based testing
python -m unittest tests/test.py

# object-based testing
python -m pytest
```

## Docs

```shell
cd docs && make html
```
