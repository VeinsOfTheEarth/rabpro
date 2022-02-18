import pandas as pd
import geopandas as gpd

import rabpro
from rabpro.basin_stats import Dataset

import seaborn as sns

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


urls, tasks = rabpro.basin_stats.compute(
    dataset_list, basins_gdf=basin, folder="rabpro"
)

tag_list = ["temperature", "precip"]
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

# ----

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

data = pd.read_csv("test.csv")

f, axs = plt.subplots(1, 1)
sns.lineplot(x="date", y="precip_mean", data=data, ax=axs)
date_form = DateFormatter("%Y-%m")
axs.xaxis.set_major_formatter(date_form)
axs.xaxis.set_major_locator(mdates.YearLocator())
plt.show()


res = gpd.GeoDataFrame(data.merge(basin, on="id"))
res = res.set_geometry("geometry")
res.to_file("res.gpkg", driver="GPKG")

