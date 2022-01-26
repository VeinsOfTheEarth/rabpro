from pathlib import Path
import ee  # earthengine-api
import numpy as np
from ee import batch
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

ee.Initialize()
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# ---- HydroBasins Level 1 ----
hybas_1 = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_1")
raw_export = batch.Export.table.toDrive(
    collection=hybas_1, description="hybas_1", fileFormat="SHP"
)
batch.Task.start(raw_export)

# pull file from GDrive
file_list = drive.ListFile({"q": "'root' in parents and trashed=false"}).GetList()
# file_list[0]['title'][0:7] == "hybas_1"
file_positions = list(
    np.where([file1["title"][0:7] == "hybas_1" for file1 in file_list])[0]
)
file_ids = [file_list[i]["id"] for i in file_positions]
file_names = [
    "Data/HydroBasins/level_one/" + file_list[i]["title"] for i in file_positions
]
[print(id, name) for id, name in zip(file_ids, file_names)]

[
    drive.CreateFile({"id": id}).GetContentFile(name)
    for id, name in zip(file_ids, file_names)
]

# ---- HydroBasins Level 12 ----
# https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_v1_Basins_hybas_12#description

hybas_12 = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")

# print(hybas_12.first().getInfo())
# print(hybas_12.filterMetadata("HYBAS_ID", "equals", 1120319380).getInfo())
# print(hybas_12.filterMetadata("HYBAS_ID", "equals", 1120319380).first().getInfo())
# print(hybas_12.filterMetadata("HYBAS_ID", "greater_than", 1000000000).filterMetadata("HYBAS_ID", "less_than", 2000000000).first().getInfo())

hybas_12 = hybas_12.filterMetadata(
    "HYBAS_ID", "greater_than", 7000000000
).filterMetadata("HYBAS_ID", "less_than", 8000000000)

raw_export = batch.Export.table.toDrive(
    collection=hybas_12, description="hybas_12", fileFormat="SHP"
)
batch.Task.start(raw_export)

file_list = drive.ListFile({"q": "'root' in parents and trashed=false"}).GetList()
file_positions = list(
    np.where([file1["title"][0:8] == "hybas_12" for file1 in file_list])[0]
)
file_ids = [file_list[i]["id"] for i in file_positions]
file_names = [
    "Data/HydroBasins/level_twelve/" + file_list[i]["title"] for i in file_positions
]
[print(id, name) for id, name in zip(file_ids, file_names)]
[
    drive.CreateFile({"id": id}).GetContentFile(name)
    for id, name in zip(file_ids, file_names)
]
