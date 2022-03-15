# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# Since we aren't installing package here, we mock imports of the dependencies.
from unittest.mock import Mock

sys.modules["appdirs"] = Mock()
sys.modules["bs4"] = Mock()
sys.modules["cv2"] = Mock()
sys.modules["ee"] = Mock()
sys.modules["gdal"] = Mock()
sys.modules["osgeo"] = Mock()
sys.modules["geopandas"] = Mock()
sys.modules["numpy"] = Mock()
sys.modules["pandas"] = Mock()
sys.modules["pyproj"] = Mock()
sys.modules["rivgraph"] = Mock()
sys.modules["rivgraph.im_utils"] = Mock()
sys.modules["scipy"] = Mock()
sys.modules["scipy.interpolate"] = Mock()
sys.modules["scipy.ndimage.morphology"] = Mock()
sys.modules["shapely"] = Mock()
sys.modules["shapely.geometry"] = Mock()
sys.modules["shapely.ops"] = Mock()
sys.modules["tqdm"] = Mock()
sys.modules["skimage"] = Mock()
sys.modules["gdown"] = Mock()

# -- Project information -----------------------------------------------------

project = "rabpro"
copyright = "2021, J. Schwenk, J. Stachelek, T. Zussman, & J. Rowland"
author = "T. Zussman, J. Schwenk, J. Stachelek, & J. Rowland"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import sphinx_rtd_theme
import rabpro

extensions = [
    "sphinx_rtd_theme",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_panels",
    "nbsphinx",
]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

autodoc_default_options = {"exclude-members": "__weakref__"}

autodoc_mock_imports = [
    "appdirs",
    "bs4",
    "cv2",
    "ee",
    "gdal",
    "geopandas",
    "numpy",
    "osgeo",
    "pandas",
    "pyproj",
    "requests",
    "rivgraph",
    "scipy",
    "shapely",
    "skimage",
    "tqdm",
    "gdown",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

panels_css_variables = {
    "tabs-color-label-active": "hsla(231, 99%, 66%, 1)",
    "tabs-color-label-inactive": "rgba(178, 206, 245, 0.62)",
    "tabs-color-overline": "rgb(207, 236, 238)",
    "tabs-color-underline": "rgb(207, 236, 238)",
    "tabs-size-label": "1rem",
}

html_show_sourcelink = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static"]

html_css_files = [
    "css/custom.css",
]
