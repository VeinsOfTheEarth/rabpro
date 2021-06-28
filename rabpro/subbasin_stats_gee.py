import os
import ee
import json
import pandas as pd
import numpy as np
import rasterstats_modified as rsm
import utils as ru

def main(sb_inc_gdf, stYYYY, enYYYY, verbose=False, folder=None):
    """Compute subbasin statistics for each raster in control file"""
    # For improved speed, rather than computing statistics for each subbasin,
    # fetch values for the subbasins, then compute statistics by combining
    # the values as we move downstream to the next subbasin

    # Dictionary for determining which rasters and statistics to compute
    control = get_controls(sb_inc_gdf.DA.values[0])
    ee.Authenticate()

    # Convert GeoDataFrame to ee.Feature objects
    features = []
    for i in range(sb_inc_gdf.shape[0]):
        geom = sb_inc_gdf.iloc[i : i + 1, :]
        jsonDict = json.loads(geom.to_json())
        geojsonDict = jsonDict["features"][0]
        features.append(ee.Feature(geojsonDict))
    featureCollection = ee.FeatureCollection(features)

    # For each raster
    for r in control:
        imgcol = ee.ImageCollection(r["rastpath"]).select(r["bands"]).filterDate(stYYYY, enYYYY)
        if verbose:
            print(f"Computing subbasin stats for {r['rastpath']}...")

        # Generate reducer - mean and count always computed
        reducer = ee.Reducer.count().combine(reducer2=ee.Reducer.mean(), sharedInputs=True)

        if "min" in r["stats"] and "max" in r["stats"]: # or "range" in r["stats"]
            reducer = reducer.combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        elif "min" in r["stats"]:
            reducer = reducer.combine(reducer2=ee.Reducer.min(len(r["bands"])), sharedInputs=True)
        elif "max" in r["stats"]:
            reducer = reducer.combine(reducer2=ee.Reducer.max(len(r["bands"])), sharedInputs=True)

        if "std" in r["stats"]:
            reducer = reducer.combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        if "sum" in r["stats"]:
            reducer = reducer.combine(reducer2=ee.Reducer.sum(), sharedInputs=True)

        pct_list = [int(pct[3:]) for pct in r["stats"] if pct[:3] == "pct"]
        if pct_list:
            reducer = reducer.combine(reducer2=ee.Reducer.percentile(pct_list), sharedInputs=True)

        def map_func(img):
            # TODO: change to reduceRegion or simplify geometries
            return img.reduceRegions(collection=featureCollection, reducer=reducer, scale=r["resolution"])

        reducedFC = imgcol.map(map_func)
        table = reducedFC.flatten()

        # Map across feature collection and use min and max to compute range
        # if "range" in r["stats"]:

        #print(table.getDownloadURL(filetype='csv'))
        task = ee.batch.Export.table.toDrive(
            collection=table, description=r["rID"], folder=folder, fileFormat="csv"
        )
        task.start()

def get_controls(DAmax):

    """Prepare paths and parameters for computing subbasin raster stats"""

    # Determine if we need to use the coarse rasters
    DAthresh = 50000  # km^2, analyses with any basin DAs larger than this value will use the coarse DEM. Testing on the Colville showed that using the coarse resulted in differences in less than 1% for all stats (most were well under 1%)
    usecoarse = DAmax > DAthresh

    # Load raster metadata file
    datapaths = ru.get_datapaths()
    rast_df = pd.read_csv(datapaths["metadata"])

    # Rid the nans (they load because of column formatting extending beyond the last row of data)
    rast_df = rast_df[~pd.isna(rast_df.dataID)]

    # Create a control dictionary for computing subbasin stats
    controlkeys = [
        "rastpath",
        "bands",
        "maskraster",
        "stats",
        "nodatavals",
        "rID",
        "rname",
        "units",
        "resolution"
    ]

    rasters = []
    for _, row in rast_df.iterrows():
        img = {}
        # Skip some of the data entries -- they are not meant to be zonal-statted
        rID = row["dataID"]

        if row["is_raster?"] == "no":
            continue

        if row["skip?"] == "yes":
            continue

        img["rID"] = rID
        img["rname"] = row["nominally"]

        # GEE path to raster
        img["rastpath"] = row["rel_path"]

        # resolution
        resolution = 0.001 if usecoarse else row["resolution"]
        img["resolution"] = resolution

        # Dates - GEE takes care of out of range dates for us

        # Bands
        bands_tmp = row["bands"]
        if pd.isna(bands_tmp):
            bands_tmp = []
        else:
            bands_tmp = list(set([band.strip() for band in bands_tmp.split(",")]))

        img["bands"] = bands_tmp

        # Should we also mask water pixels?
        if row["water mask?"] == "yes":
            img["maskraster"] = datapaths["watermask"] #TODO
        else:
            img["maskraster"] = None

        # Which stats to compute?
        stats_tmp = row["stats"]
        if pd.isna(stats_tmp):
            stats_tmp = []
        else:
            stats_tmp = [st.strip() for st in stats_tmp.split(",")]
        # Count and mean are always returned
        stats_tmp = stats_tmp + ["count", "mean"]
        stats_tmp = list(set(stats_tmp))  # uniquify list
        img["stats"] = stats_tmp

        # nodatavals?
        if pd.isna(row["nodata"]) or row["nodata"].lower() == "none":
            img["nodatavals"] = None
        else:
            img["nodatavals"] = row["nodata"]

        # units
        img["units"] = row["units"]
        
        rasters.append(img)

    return rasters
