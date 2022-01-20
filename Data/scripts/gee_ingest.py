import os
import config
import subprocess

filename = os.path.expanduser(config.filename)
out_folder = os.path.expanduser(config.out_folder)
os.makedirs(out_folder, exist_ok=True)
nlayers = config.nlayers


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


#  ---- Execute
pull_tifs_from_nc(filename, out_folder, nlayers)
