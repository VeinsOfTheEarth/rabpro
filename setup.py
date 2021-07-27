import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rabpro",
    version="0.2",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jschwenk/rabpro",
    project_urls={
        "Bug Tracker": "https://github.com/jschwenk/rabpro/issues",
    },
    install_requires=[
        "python>=3.6",
        "gdal",
        "numpy",
        "geopandas>=0.7.0",
        "scikit-image",
        "opencv",
        "matplotlib",
        "pyproj",
        "shapely",
        "rivgraph>=0.3",
        "requests",
        "appdirs",
        "earthengine-api",
        "tqdm",
        "beautifulsoup4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages()
)
