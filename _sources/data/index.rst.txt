.. _data:

============
GEE and Data
============

RaBPro requires setting up Google Earth Engine and downloading some data files
in order to run. This section describes those requirements, along with some
additional options.

Google Earth Engine (GEE)
-------------------------

Authentication
~~~~~~~~~~~~~~
In order to use RaBPro, you'll need to sign up for a free GEE account `here
<https://signup.earthengine.google.com/#!/>`__. Once you've been approved and
installed the GEE Python API (typically installed as a RaBPro dependency), you
can use the GEE CLI to obtain a credential token by running ``earthengine
authenticate`` and following the instructions. More information can be found at
the `GEE Python install page
<https://developers.google.com/earth-engine/guides/python_install>`_ and the
`GEE CLI page
<https://developers.google.com/earth-engine/guides/python_install>`_.

Datasets
~~~~~~~~
RaBPro uses a JSON file containing metadata for all datasets available through
GEE in order to validate requests to the GEE API and fail fast if they're
invalid. This JSON file is updated daily server-side. RaBPro caches a local copy
of this file, and will attempt to retrieve an updated version when run or daily
(the less frequent of the two). The file can be found `here
<https://github.com/VeinsOfTheEarth/rabpro/blob/main/Data/gee_datasets.json>`__, and
will be downloaded to the following locations:

Linux: ``~/.local/share/rabpro/``

macOS: ``~/Library/Application Support/rabpro/``

Windows: ``%UserProfile%\AppData\Local\jschwenk\rabpro\``

This file should not be edited manually - any changes will be overwritten when
an updated version is retrieved.

User Datasets
~~~~~~~~~~~~~

.. csv-table:: Datasets included in the user data catalog:
   :file: ../user_gee_datasets.csv
   :align: center

You can add custom datasets to GEE by following the instructions `here
<https://developers.google.com/earth-engine/guides/image_upload>`__. To ensure
compatibility with RaBPro, upload GeoTIFFs and specify a no-data value if
applicable. In order to use this dataset with RaBPro follow the instructions in
`User Dataset Configuration`_.

Local Data
----------

MERIT and HydroBASINS
~~~~~~~~~~~~~~~~~~~~~
RaBPro requires local copies of the `MERIT Hydro
<http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_, and `HydroBASINS
<https://www.hydrosheds.org/downloads>`_ datasets to delineate watersheds and
compute elevation profiles.

Specifically, RaBPro needs level one and level twelve shapefiles from
HydroBASINS, and all MERIT tif files for a given tile. These can be installed
manually or programatically through the RaBPro API or CLI.

The data files should be located in the following paths:

Linux: ``~/.local/share/rabpro/``

macOS: ``~/Library/Application Support/rabpro/``

Windows: ``%UserProfile%\AppData\Local\rabpro\rabpro\``

with the following directory structure:
::
    MERIT_Hydro/
    ├─ MERIT_ELEV_HP/
    ├─ MERIT_FDR/
    ├─ MERIT_UDA/
    ├─ MERIT_WTH/
    ├─ MERIT103/
    HydroBasins/
    ├─ level_one/
    ├─ level_twelve/


User Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Place (or create) a file `user_gee_datasets.json` at the location returned by: 

.. code-block:: python

        from rabpro import data_utils as du
        du._path_generator_util(None, None)[1]

User datasets typically do not have "bands". To create a valid json entry for such bandless datasets, enter a json block similar to the following:

.. code-block:: json

        [
            {
                "id": "projects/sat-io/open-datasets/Geomorpho90m/slope",
                "name": "Geomorpho90m: Slope",
                "start_date": null,
                "end_date": null,
                "type": "image_collection",
                "bands": {
                    "None": {
                        "resolution":90
                    }
                }
            }
        ]

Call this dataset from `subbasin_stats` without specifying a band:

.. code-block:: python

    data, task = rabpro.subbasin_stats.main(
        [
            Dataset(
                "projects/sat-io/open-datasets/Geomorpho90m/slope", time_stats=["median"]
            )
        ],
        gee_feature_path="your/asset/path",
    )
