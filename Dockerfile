FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
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

# virtualenv for honeybee
RUN git clone https://github.com/kakaobrain/honeybee.git &&\
    cd honeybee &&\
    python -m venv honeybee_venv &&\
    honeybee_venv/bin/pip install --upgrade pip setuptools wheel &&\
    honeybee_venv/bin/pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 &&\
    honeybee_venv/bin/pip install -r requirements.txt &&\
    honeybee_venv/bin/pip install -r requirements_demo.txt &&\
    honeybee_venv/bin/pip install flask sentencepiece opencv-python

# download honeybee checkpoints
RUN mkdir -p honeybee/checkpoints &&\
    wget -q https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M144.tar.gz -O honeybee/checkpoints/7B-C-Abs-M144.tar.gz &&\
    tar -xvzf honeybee/checkpoints/7B-C-Abs-M144.tar.gz -C honeybee/checkpoints &&\
    rm -f honeybee/checkpoints/7B-C-Abs-M144.tar.gz
RUN wget -q https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-D-Abs-M144.tar.gz -O honeybee/checkpoints/7B-D-Abs-M144.tar.gz &&\
    tar -xvzf honeybee/checkpoints/7B-D-Abs-M144.tar.gz -C honeybee/checkpoints &&\
    rm -f honeybee/checkpoints/7B-D-Abs-M144.tar.gz
RUN wget -q https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M256.tar.gz -O honeybee/checkpoints/7B-C-Abs-M256.tar.gz &&\
    tar -xvzf honeybee/checkpoints/7B-C-Abs-M256.tar.gz -C honeybee/checkpoints &&\
    rm -f honeybee/checkpoints/7B-C-Abs-M256.tar.gz

# virtualenv for llava
RUN git clone https://github.com/haotian-liu/LLaVA.git &&\
    cd LLaVA &&\
    python -m venv llava_venv &&\
    llava_venv/bin/pip install --upgrade pip setuptools wheel &&\
    llava_venv/bin/pip install -e . &&\
    llava_venv/bin/pip install flask opencv-python protobuf

# remove pip cache
RUN rm -rf /root/.cache/pip

COPY scripts/honeybee_server.py /root/honeybee/server.py
COPY scripts/llava_server.py /root/LLaVA/server.py
COPY scripts/entrypoint.sh /root/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/root/entrypoint.sh"]
