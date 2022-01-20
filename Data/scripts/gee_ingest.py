import os
import glob
import rabpro
import config
import subprocess

filename = os.path.expanduser(config.filename)
out_folder = os.path.expanduser(config.out_folder)
os.makedirs(out_folder, exist_ok=True)
nlayers = config.nlayers

title = config.title
description = config.description
citation = config.citation
time_start = config.time_start
epsg = config.epsg

gee_user = config.gee_user
gcp_bucket = config.gcp_bucket
gcp_folder = config.gcp_folder


def get_layer(filename, out_folder, layer):
    """get_layer(filename, 1)
    """
    out_path = "{}{}_{}.tif".format(out_folder, os.path.basename(filename), layer)
    shell_cmd = "gdal_translate -of GTiff -ot Float32 -co COMPRESS=DEFLATE -co TILED=YES -b {} {} {}".format(
        layer, filename, out_path
    )

    if not os.path.exists(out_path):
        print(shell_cmd)
        subprocess.call(shell_cmd)

    return out_path


def pull_tifs_from_nc(filename, out_folder, nlayers):
    for i in range(0, nlayers):
        get_layer(filename, out_folder, i + 1)

    return out_folder


def push_tifs(out_folder):
    tif_list = glob.glob(out_folder + "*")
    # tif = tif_list[0]
    rabpro.utils.upload_gee_tif_asset(
        tif,
        gee_user,
        gcp_bucket,
        title,
        gcp_folder=gcp_folder,
        description=description,
        citation=citation,
        time_start=time_start,
        epsg=epsg,
    )
    return None


#  ---- Execute
if os.path.splitext(filename)[1] == ".nc":
    pull_tifs_from_nc(filename, out_folder, nlayers)

push_tifs(out_folder)
