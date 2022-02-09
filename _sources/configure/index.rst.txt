.. _configure:

=============
Configuration
=============

Data
----

Basin delineation
~~~~~~~~~~~~~~~~~

To use the rabpro utilities for delineating subbasins, you'll need to download MERIT tiles (recommended for small basins or high resolution applications) or HydroBASINS (recommended for very large basins or low resolution applications).

Before initiating either of these downloads, you may want to configure the path where rabpro expects to find this data. Perhaps you want to change the default (set according to the `appdirs <https://github.com/ActiveState/appdirs>`_ package) to a particular project folder or disk drive. To do this, set the ``RABPRO_DATA`` environment variable either in an activate python session with ``os.environ`` or on a persistent basis through your operating system.

Downloading HydroBASINS
_______________________

To download level 1 HydroBASINS, execute the following commandline call:

.. code-block:: shell

        rabpro download hydrobasins

Downloading MERIT
_________________

To download MERIT, you'll need to request a username and password on the MERIT `homepage <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_.

Unless you want to download the full global extent, you'll probably need to identify specific MERIT "tiles" of interest. You can do this following the logic in the `MERIT Hydro
<http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_ documentation or use the rabpro function ``rabpro.utils.coords_to_merit_tile`` to obtain a tile identifier (e.g. n30e150) which gets passed to the commandline ``rabpro download merit`` utility:

.. code-block:: shell

         rabpro download merit n30e150 <username> <password>

Subbasin statistics
~~~~~~~~~~~~~~~~~~~

By default, rabpro comes enabled to work with all of the raster assets in the `public GEE data catalog <https://developers.google.com/earth-engine/datasets/>`_. It also is enabled to work with select "user" assets listed below:

.. csv-table:: Datasets included in the user data catalog:
   :file: ../user_gee_datasets.csv
   :align: center

You can request that a user asset be added to this list by filing an `issue <https://github.com/VeinsOfTheEarth/rabpro/issues/new?assignees=&labels=data+request&template=data-request.yml>`_.

Software
--------

Subbasin statistics
~~~~~~~~~~~~~~~~~~~

To use rabpro utilities for pulling subbasins statistics from Google 
Earth Engine (GEE), you'll need to sign up for a free GEE account `here
<https://signup.earthengine.google.com/#!/>`__. Once you've been approved and
installed the GEE Python API (typically installed as a rabpro dependency), you
can use the GEE CLI to obtain a credential token by running ``earthengine
authenticate`` and following the instructions. More information can be found at
the `GEE Python install page
<https://developers.google.com/earth-engine/guides/python_install>`_ and the
`GEE CLI page
<https://developers.google.com/earth-engine/guides/python_install>`_.

If you are working with complex or large numbers of subbasin polygons (or want to 
upload your own raster assets), you may be interested in the rabpro utilities for
programmatic GEE asset uploads. These utilities require a writeable Google Cloud 
Platform (GCP) bucket as well as installation and authentication for the gsutil program. We recommend installing from the Python package as described `here <https://cloud.google.com/storage/docs/gsutil_install#expandable-2>`_.

