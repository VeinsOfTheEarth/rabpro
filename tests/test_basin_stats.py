import geopandas as gpd
import rabpro
from rabpro.subbasin_stats import Dataset
from shapely.geometry import box
import pandas as pd
import ee
import copy


coords_file = gpd.read_file(r"tests/data/Big Blue River.geojson")
gdf = gpd.GeoDataFrame(
    {"idx": [1], "geometry": [box(*coords_file.total_bounds)]}, crs="EPSG:4326"
)


def clean_res(feature):
    res = pd.DataFrame(feature["properties"], index=[0])
    res["id"] = feature["id"]
    return res


def clean_freqhist(feature, name_category):
    feature = copy.deepcopy(feature)
    res_hist = pd.DataFrame(feature["properties"]["histogram"], index=[0])
    res_hist.columns = [name_category + "_" + x for x in res_hist.columns]

    del feature["properties"]["histogram"]
    res = pd.DataFrame(feature["properties"], index=[0])
    res["id"] = feature["id"]

    res = pd.concat([res, res_hist], axis=1)

    return res


def test_customreducer():
    def asdf(feat):
        return feat.getNumber("max")

    data, task = rabpro.subbasin_stats.main(
        gdf,
        [Dataset("JRC/GSW1_3/YearlyHistory", "waterClass", stats=["max"])],
        reducer_funcs=[asdf],
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data["features"]])

    assert all(res["asdf"] == res["max"])


def test_categorical_imgcol():
    data, task = rabpro.subbasin_stats.main(
        gdf,
        [Dataset("MODIS/006/MCD12Q1", "LC_Type1", stats=["freqhist"])],
        test=True,
    )

    res = pd.concat(
        [clean_freqhist(feature, "LC_Type1") for feature in data["features"]]
    )

    assert res.shape[1] > 4


def test_timeindexed_imgcol():

    data, task = rabpro.subbasin_stats.main(
        gdf,
        [
            Dataset(
                "JRC/GSW1_3/YearlyHistory",
                "waterClass",
            )
        ],
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data["features"]])

    assert res["mean"].iloc[0] > 0
    assert res.shape[0] > 0


def test_timeindexedspecific_imgcol():

    data, task = rabpro.subbasin_stats.main(
        gdf,
        [
            Dataset(
                "JRC/GSW1_3/YearlyHistory",
                "waterClass",
                start="2017-01-01",
                end="2019-01-01",
            )
        ],
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data["features"]])

    assert res.shape[0] == 2


def test_nontimeindexed_imgcol():

    data, task = rabpro.subbasin_stats.main(
        gdf,
        [
            Dataset(
                "JRC/GSW1_3/MonthlyRecurrence",
                "monthly_recurrence",
            )
        ],
        test=True,
    )

    res = pd.concat([clean_res(feature) for feature in data["features"]])

    assert res.shape[0] > 0


def test_img():

    data, task = rabpro.subbasin_stats.main(
        gdf,
        [
            Dataset(
                "JRC/GSW1_3/GlobalSurfaceWater",
                "occurrence",
                stats=["min", "max", "range", "std", "sum", "pct50", "pct3"],
            )
        ],
        test=True,
    )

    res = pd.DataFrame(data["features"][0]["properties"], index=[0])

    assert float(res["mean"]) > 0
    assert res.shape[1] == 9
