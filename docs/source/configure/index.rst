.. _configure:

=============
Configuration
=============

Data
----

Basin delineation
~~~~~~~~~~~~~~~~~

To use the rabpro utilities for delineating basins, you'll need to download MERIT tiles (recommended for small basins or high resolution applications) or HydroBASINS (recommended for very large basins or low resolution applications).

Before initiating either of these downloads, you may want to configure the path where rabpro expects to find this data. rabpro uses the `appdirs <https://github.com/ActiveState/appdirs>`_ package to specify a default data location. You can identify the locations where rabpro expects data to exist:

.. code-block:: python

        from rabpro import utils
        datapaths = utils.get_datapaths()
        print(datapaths['root']) # Highest-level directory rabpro is looking for
        print(datapaths['MERIT_root']) 
        print(datapaths['HydroBasins_root'])


If you would like to specify a different ``root`` location, you may set the ``RABPRO_DATA`` environment variable in an active python session:

.. code-block:: python

        import os
        os.environ['RABPRO_DATA'] = 'path/to/rabpro/data'


You may also set this environment variable more permanently by adding ``RABPRO_DATA`` to your operating system's environment variables, but you may run into issues with your python interpreter not reading the OS environment variables correctly.



Downloading HydroBASINS
_______________________

To download HydroBASINS, execute the following python code:

.. code-block:: python

        from rabpro import data_utils
        data_utils.download_hydrobasins()

If `rabpro` cannot automatically fetch this data, it will print the URL of the zipped HydroBASINS file and the directory to which you should unzip this file.

Downloading MERIT-Hydro
_______________________

To download MERIT-Hydro, you'll need to request a username and password on the MERIT-Hydro `homepage <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_.

Unless you want to download the full global extent, you'll probably need to identify specific MERIT-Hydro "tiles" of interest. You can do this following the logic in the `MERIT Hydro
<http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_ documentation or use the rabpro function ``rabpro.utils.coords_to_merit_tile`` to obtain a tile identifier (e.g. n30e150) which you can pass to the ``download_merit_hydro`` function:

.. code-block:: python
	# To identify a tile 
        from rabpro import utils
        coords = (-97.355, 45.8358) # lon, lat
        utils.coords_to_merit_tile(coords)
        # Should output '"n45w100"'

.. code-block:: python
	# To download the tile
        from rabpro import data_utils
        data_utils.download_merit_dem("n45w100", your_username, your_password)

Basin statistics
~~~~~~~~~~~~~~~~~~~

By default, rabpro comes enabled to work with all of the raster assets in the `public GEE data catalog <https://developers.google.com/earth-engine/datasets/>`_. It also is enabled to work with select "user" assets listed below:

.. csv-table:: Datasets included in the user data catalog:
   :file: ../user_gee_datasets.csv
   :align: center

You can request that a user asset be added to this list by filing an `issue <https://github.com/VeinsOfTheEarth/rabpro/issues/new?assignees=&labels=data+request&template=data-request.yml>`_.

Software
--------

Basin statistics
~~~~~~~~~~~~~~~~~~~

To use rabpro utilities for pulling basin statistics from Google 
Earth Engine (GEE), you'll need to sign up for a free GEE account `here
<https://signup.earthengine.google.com/#!/>`__. Once you've been approved and
installed the GEE Python API (typically installed as a rabpro dependency), you
can use the GEE CLI to obtain a credential token by running ``earthengine
authenticate`` and following the instructions. More information can be found at
the `GEE Python install page
<https://developers.google.com/earth-engine/guides/python_install>`_ and the
`GEE CLI page
<https://developers.google.com/earth-engine/guides/python_install>`_.

If you are working with complex or large numbers of watershed basin polygons (or want to 
upload your own raster assets), you may be interested in the rabpro utilities for
programmatic GEE asset uploads. These utilities require a writeable Google Cloud 
Platform (GCP) bucket as well as installation and authentication for the ``gsutil`` program. We recommend installing from the Python package as described `here <https://cloud.google.com/storage/docs/gsutil_install#expandable-2>`_.

