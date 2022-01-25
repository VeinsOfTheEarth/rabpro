import os
import glob
import rabpro
import config
import subprocess
import datetime
import dateutil

dry_run = config.dry_run
gcp_upload = config.gcp_upload
gee_upload = config.gee_upload

filename = os.path.expanduser(config.filename)
out_folder = os.path.expanduser(config.out_folder)
os.makedirs(out_folder, exist_ok=True)
nlayers = config.nlayers

title = config.title
description = config.description
citation = config.citation
time_start = config.time_start
time_end = config.time_end
time_frequency = config.time_frequency
epsg = config.epsg

gee_user = config.gee_user
gcp_bucket = config.gcp_bucket
gcp_folder = config.gcp_folder
gee_folder = config.gee_folder

# def plus_1(x):
#     return x + 1
# list(map(plus_1, [1, 2]))


def get_layer(filename, out_folder, layer, out_path=None, dry_run=False):
    """get_layer(filename, out_folder, 1)"""
    if out_path is None:
        out_path = "{}{}_{}.tif".format(out_folder, os.path.basename(filename), layer)

    shell_cmd = "gdal_translate -of GTiff -ot Float32 -co COMPRESS=DEFLATE -co TILED=YES -b {} {} {}".format(
        layer, filename, out_path
    )

    print(shell_cmd)
    if not os.path.exists(out_path):
        if not dry_run:
            subprocess.call(shell_cmd)

    return out_path


def pull_tifs_from_nc(
    filename, out_folder, nlayers, time_start=None, time_frequency=None, dry_run=False
):

    if time_frequency == "years":
        year_start = int(time_start[0:4])
        year_end = year_start + nlayers
        out_paths = [
            "{}{}.tif".format(out_folder, str(path))
            for path in list(range(year_start, year_end + 1))
        ]

    if time_frequency == "months":
        month_stamps = []
        i = 0
        while i <= (nlayers - 1):
            month_stamp = (
                datetime.datetime.strptime(time_start, "%Y-%m-%d")
                + dateutil.relativedelta.relativedelta(months=i)
            ).strftime("%Y-%m")
            month_stamps.append(month_stamp)
            i = i + 1

        out_paths = ["{}/{}.tif".format(out_folder, str(path)) for path in month_stamps]

    for i, opath in zip(range(0, nlayers), out_paths):
        get_layer(filename, out_folder, i + 1, opath, dry_run=dry_run)

    return out_folder


def push_tifs(out_folder, **kwargs):
    tif_list = glob.glob(out_folder + "/*")
    # tif = tif_list[0]
    for tif in tif_list:
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
            gee_folder=gee_folder,
            dry_run=dry_run,
            gcp_upload=gcp_upload,
            gee_upload=gee_upload,
        )
    return None


#  ---- Execute
if os.path.splitext(filename)[1] == ".nc":
    pull_tifs_from_nc(
        filename, out_folder, nlayers, time_start, time_frequency, dry_run
    )

push_tifs(
    out_folder,
    gee_folder=gee_folder,
    dry_run=dry_run,
    gcp_upload=gcp_upload,
    gee_upload=gee_upload,
)
