.. _install:

============
Installation
============

As `rabpro` relies on a large stack of geospatial dependencies, we recommend using Anaconda to install these in a fresh environment. Otherwise you will need to undertake a manual install of GDAL, which can be quite complex.

::

   $ conda create -n rabpro python=3.9 geopandas -c conda-forge
   $ conda activate rabpro

Once that base is in place, you can proceed with installing either from Anaconda or pip.

Installation via *conda*
------------------------

The latest 'stable' version of *rabpro* can be installed using `conda`.

::

   $ conda install -c jschwenk rabpro

This will install rabpro and all of its dependencies, as listed in
`environment.yml
<https://github.com/VeinsOfTheEarth/rabpro/blob/master/environment.yml>`_.

Installation from source
------------------------

If you would prefer to install bleeding edge *rabpro* from source or you are not using conda-installed GDAL, do the following:

1. Clone the repository
::

   $ git clone https://github.com/VeinsOfTheEarth/rabpro.git

or, if you would prefer to use an SSH key:

::

   $ git clone git@github.com:VeinsOfTheEarth/rabpro.git

2. Install dependencies

Assuming you've activated your fresh `rabpro` environment:
::

   $ conda env update -f environment.yml

3. Local installation using `setuptools`

In the cloned folder, run the following:
::

   $ pip install -e .

