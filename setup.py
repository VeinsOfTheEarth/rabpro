import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rabpro",
    version="0.3.0",
    author="Example Author",
    author_email="author@example.com",
    description="Package to delineate subbasins and compute statistics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VeinsOfTheEarth/rabpro",
    project_urls={"Bug Tracker": "https://github.com/VeinsOfTheEarth/rabpro/issues",},
    scripts=["rabpro/cli/rabpro"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)
