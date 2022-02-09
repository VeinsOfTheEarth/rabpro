.. rabpro documentation master file, created by
   sphinx-quickstart on Fri Jul 16 16:08:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rabpro's documentation!
==================================

rabpro provides methods to compute subbasin geometries from coordinate pairs, and compute statistics over these subbasins using `Google Earth Engine <https://developers.google.com/earth-engine/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Setup:
   :hidden:
   
   install/index
   configure/index   

.. toctree::
   :maxdepth: 2
   :caption: Examples:
   :hidden:
   :glob:
      
   examples/notebooks/basic_example.ipynb
   examples/notebooks/full_example.ipynb
   examples/notebooks/basin_stats.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Miscellanea:
   :hidden:

   contributing/index
   issues/index
   references/index
   apiref/index


.. panels::

   :card: + text-center bg-secondary
   .. link-button:: install/index
        :type: ref
        :text: Installation
        :classes: btn-link stretched-link font-weight-bold
   ---    
   :card: + text-center bg-secondary
   .. link-button:: configure/index
        :type: ref
        :text: Configuration
        :classes: btn-link stretched-link font-weight-bold
   ---
   :card: + text-center bg-secondary
   .. link-button:: examples/notebooks/basic_example
        :type: ref
        :text: Basic example
        :classes: btn-link stretched-link font-weight-bold
   ---
   :card: + text-center bg-secondary
   .. link-button:: examples/notebooks/full_example
        :type: ref
        :text: Full example
        :classes: btn-link stretched-link font-weight-bold
   ---
   :column: col-lg-12 p-2
   :card: + text-center bg-info
   .. link-button:: apiref/index
        :type: ref
        :text: API Reference
        :classes: btn-link stretched-link font-weight-bold