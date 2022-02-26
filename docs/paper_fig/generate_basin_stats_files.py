# ---- setup ----
import rioxarray
import pandas as pd
import geopandas as gpd
import os

import rabpro
from rabpro.basin_stats import Dataset

path_base = "docs/paper_fig/"
path_base = r'X:\Research\RaBPro\Code\docs\paper_fig'
basin = gpd.read_file(os.path.join(path_base, "basin_merit.gpkg"))
basin["id"] = 0

dataset_list = [
    Dataset(
        "ECMWF/ERA5_LAND/MONTHLY",
        "temperature_2m",
        stats=["mean"],
        start="2000-06-01",
        end="2021-09-01",
    ),
    Dataset(
        "NASA/GPM_L3/IMERG_MONTHLY_V06",
        "precipitation",
        stats=["mean"],
        start="2000-06-01",
        end="2021-09-01",
    ),
    Dataset(
        "JRC/GSW1_3/GlobalSurfaceWater",
        "occurrence",
        stats=["mean"],
    ),
    Dataset(
        "projects/sat-io/open-datasets/Geomorpho90m/slope",
        stats=["mean"],
        mosaic=True
    ),
    Dataset(
        "MODIS/006/MOD13A2",
        "NDVI",
        stats=["mean"],
        start="2000-06-01",
        end="2021-09-01",
    ),
    Dataset(
        "CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density",
        'unwpp-adjusted_population_density'
    ),
    Dataset(
        "NASA_USDA/HSL/SMAP10KM_soil_moisture",
        'ssm'
    ),

]


tag_list = ["temperature", "precip", 'occurrence', 'slope', 'ndvi', 'pop_density', 'soil_moisture']
fnames = [path_base + tag + ".tif" for tag in tag_list]

# ---- pull stats ----
urls, tasks = rabpro.basin_stats.compute(
    dataset_list, basins_gdf=basin, folder="rabpro"
)

data = rabpro.basin_stats.fetch_gee(urls, tag_list)

# format time column
data = data.rename(columns={"temperature_system:index": "date"}).assign(
    year=lambda x: x.date.str.slice(0, 4),
    month=lambda x: x.date.str.slice(4, 6),
    day="01",
)
date_column = pd.to_datetime(data[["year", "month", "day"]])
data["date"] = date_column
data = data.drop(["year", "month", "day"], axis=1).filter(regex="^((?!system).)*$")

data.to_csv(os.path.join(path_base, "test.csv"), index=False)

# ---- plot stats ----
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(os.path.join(path_base,"test.csv"))
data["date"] = pd.to_datetime(data["date"])


plt.close()
sf = 1.3
sns.set(rc={'figure.figsize':(3.3*sf,1.59 * sf)})
sns.set_style('darkgrid')
ax = sns.lineplot(data['date'], data['precip_mean'])
ax.set(xlabel='', ylabel='Precipitation (mm/hr)')
plt.tight_layout()

plt.close()
sns.set(rc={'figure.figsize':(3.3*sf,1.59 * sf)})
sns.set_style('darkgrid')
ax = sns.lineplot(data['date'], data['temperature_mean']-273, color='lightcoral')
ax.set(xlabel='', ylabel='Temperature (C)')
plt.tight_layout()


# ---- pull images ----

urls, tasks = rabpro.basin_stats.image(dataset_list, basins_gdf=basin)
res = [rabpro.basin_stats._fetch_raster(url, fname) for url, fname in zip(urls, fnames)]

# ---- generate clipped images ----


def img_clip(fpath, gdf):
    r_clip = rioxarray.open_rasterio(fpath, masked=True)
    r_clip = r_clip.rio.clip(gdf.geometry, gdf.crs, drop=False)
    r_clip.rio.to_raster(fpath.replace(".tif", "_clip.tif"))
    return r_clip


[img_clip(fpath, basin) for fpath in fnames]

# ---- plot images ----

# rasterio.plot.show(res)
# plt.imshow(res)
# plt.show()
