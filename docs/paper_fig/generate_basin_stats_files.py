# ---- setup ----
import rioxarray
import pandas as pd
import geopandas as gpd

import rabpro
from rabpro.basin_stats import Dataset

path_base = "docs/paper_fig/"
basin = gpd.read_file(path_base + "basin_merit.gpkg")
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
]

tag_list = ["temperature", "precip"]

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

data.to_csv("test.csv", index=False)

# ---- plot stats ----
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("test.csv")
data["date"] = pd.to_datetime(data["date"])

f, axs = plt.subplots(1, 1)
fig_precip = sns.lineplot(x="date", y="precip_mean", data=data, ax=axs)
fig_precip.set(xlabel="", ylabel="Precipitation (mm/hr)")
plt.show()

# ---- pull images ----

urls, tasks = rabpro.basin_stats.image(dataset_list, basins_gdf=basin)
fnames = [path_base + tag + ".tif" for tag in tag_list]
res = [rabpro.basin_stats._fetch_raster(url, fname) for url, fname in zip(urls, fnames)]

# ---- plot images ----

r_reclass_clip = rioxarray.open_rasterio(fnames[0], masked=True)
# r_reclass_clip.rio.write_crs("epsg:4326", inplace=True)
# r_reclass_clip = r_reclass_clip.rio.clip(gdf.geometry, gdf.crs, drop=False)

# rasterio.plot.show(res)
# plt.imshow(res)
# plt.show()
