FROM continuumio/miniconda3:4.7.10

LABEL "repository"="https://github.com/veinsoftheearth/rabpro"
LABEL "maintainer"="Jemma Stachelek <stachel2@msu.edu>"

RUN apt update --allow-releaseinfo-change
RUN apt install -y build-essential libglu1-mesa-dev freeglut3-dev mesa-common-dev

RUN conda install -y anaconda-client conda-build

VOLUME ["/rabpro"]
WORKDIR /rabpro/conda
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash"]
