ARG CUDA_VERSION=11.8 # or 12.1
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt -o Acquire::AllowInsecureRepositories=true update \
    && apt-get install -y \
    curl \
    wget \
    build-essential \
    git \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Install dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-distutils python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python to point to python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# set workdir root
WORKDIR /root

# honeybee setup
COPY honeybee_setup.sh /root/honeybee_setup.sh
RUN /bin/bash /root/honeybee_setup.sh

# LLaVA setup
COPY llava_setup.sh /root/llava_setup.sh
RUN /bin/bash /root/llava_setup.sh

# remove pip cache
RUN rm -rf /root/.cache/pip

COPY scripts/honeybee_server.py /root/honeybee/server.py
COPY scripts/llava_server.py /root/LLaVA/server.py
COPY scripts/entrypoint.sh /root/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/root/entrypoint.sh"]
