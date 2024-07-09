# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04
ENV TZ=UTC

# Install necessary packages
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends --fix-missing \
    apt-utils \
    build-essential \
    bash \
    wget \
    git \
    curl \
    ssh \
    subversion \
    tar \
    unzip \
    patch \
    gzip \
    bzip2 \
    file \
    gnupg \
    coreutils \
    mercurial \
    nano \
    pkg-config \
    tree \
    flex \
    automake \
    make \
    cmake \
    cmake-curses-gui \
    python3.11 python3-pip cxxtest libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install OpenBLAS
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring.gpg && \
    mv cuda-keyring.gpg /usr/share/keyrings/ && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-11-2


# Clone and install libxsmm
RUN git clone https://github.com/hfp/libxsmm.git && \
    cd libxsmm && \
    make && \
    make install PREFIX=/usr/local

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Install the application
RUN pip install . --break-system-packages

