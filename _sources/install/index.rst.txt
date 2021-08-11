.. _install:

=======
Install
=======

Installation via *conda*
------------------------

The latest 'stable' version of *RaBPro* can be installed using `conda`. In order
to avoid dependency conflicts, we recommend installing *RaBPro* into a fresh
`conda` environment. Open Terminal (Mac/Unix) or Anaconda Prompt (Windows) and
type:
::

   $ conda env create rabpro
   $ conda activate rabpro
   $ conda install -c jschwenk rabpro

This will install all of rabpro's dependencies as well, as listed in
`environment.yml
<https://github.com/jonschwenk/rabpro/blob/master/environment.yml>`_.

Installation from source
------------------------

If you would prefer to install *RaBPro* from source, do the following:

1. Clone the repository
::

   $ git clone https://github.com/jonschwenk/rabpro.git

or, if you would prefer to use an SSH key:

::

   $ git clone git@github.com:jonschwenk/rabpro.git

2. Install dependencies

Create a new `conda` environment from `environment.yml
<https://github.com/jonschwenk/rabpro/blob/master/environment.yml>`_:
::

   $ conda env create --file environment.yml

3. Local installation using `setuptools`

In the cloned folder, run the following:
::

   $ pip install -e .

or

::

   $ python setup.py install

