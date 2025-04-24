FROM ubuntu:22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    TZ=Europe/London \
    PYTHON_VERSION=3.10 \
    OPENFOAM_VERSION=10

# Install system libraries and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    libosmesa6-dev \
    tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install OpenFOAM
RUN wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc && \
    add-apt-repository "http://dl.openfoam.org/ubuntu" && \
    apt-get update && \
    apt-get install -y --no-install-recommends openfoam${OPENFOAM_VERSION} && \
    sed -i 's|export ParaView_GL=mesa|export ParaView_GL=system|' /opt/openfoam${OPENFOAM_VERSION}/etc/config.sh/paraview

# Install ParaView
WORKDIR /opt
RUN wget "https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.12&type=binary&os=Linux&downloadFile=ParaView-5.12.0-MPI-Linux-Python3.10-x86_64.tar.gz" -O paraview.tar.gz && \
    tar -xzf paraview.tar.gz && \
    mv ParaView-5.12.0-MPI-Linux-Python3.10-x86_64 paraview && \
    rm paraview.tar.gz

# Install Python
RUN add-apt-repository universe && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-tk \
    python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    python3 -m pip install --upgrade pip
 
# Install Python dependencies
RUN pip install \
    matplotlib==3.8.2 \
    motioncapture==1.0a2 \
    mujoco==3.2.2 \
    numpy==1.23.3 \
    scipy==1.10.1 \
    sphinx==8.0.2 \
    furo==2024.8.6 \
    "skyc_utils @ git+https://github.com/AIMotionLab-SZTAKI/skyc_utils.git" \
    opencv-python==4.10.0.84 \
    cyipopt==1.5.0 \
    casadi==3.6.4 \
    pyyaml==6.0.2 \
    pandas \
    networkx \
    trimesh \
    manifold3d \
    rtree

WORKDIR /home/app

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
    rm -rf /opt/openfoamparaview510
        
CMD ["/bin/bash", "-c", "source /opt/openfoam10/etc/bashrc && pip install -e ./ > /dev/null 2>&1 && exec bash"]
