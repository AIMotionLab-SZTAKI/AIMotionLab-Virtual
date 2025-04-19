FROM ubuntu:22.04

# Install system essential system libraries
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    gnupg2 \
    ca-certificates \
    lsb-release \
    git \
    build-essential \
    pkg-config \
    coinor-libipopt-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglew-dev \
    xorg-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxcursor-dev \
    libosmesa6 \
    libosmesa6-dev

# Install python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python3 -m pip install --upgrade pip

# Install project source
RUN git clone https://github.com/szabo-krisztian/AIMotionLab-Virtual.git /home/AIMotionLab-Virtual

WORKDIR /home/AIMotionLab-Virtual

RUN git checkout origin/rework_windflow

RUN pip install --upgrade pip && pip install -e .

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Hungary/Budapest

RUN apt-get install -y tzdata && ln -fs /usr/share/zoneinfo/Europe/London /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

RUN sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"

RUN add-apt-repository http://dl.openfoam.org/ubuntu

RUN apt-get update

RUN apt-get -y install openfoam10

RUN sed -i 's|export ParaView_GL=mesa|export ParaView_GL=system|' /opt/openfoam10/etc/config.sh/paraview

RUN rm -rf /var/lib/apt/lists/* 
