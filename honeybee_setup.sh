#!/usr/bin/env sh

# virtualenv for honeybee
git clone https://github.com/kakaobrain/honeybee.git &&\
    cd honeybee &&\
    python -m venv honeybee_venv &&\
    honeybee_venv/bin/pip install --upgrade pip setuptools wheel &&\
    if [ "$CUDA_VERSION" = "11.8" ]; then\
        honeybee_venv/bin/pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118;\
    elif [ "$CUDA_VERSION" = "12.1" ]; then\
        honeybee_venv/bin/pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121;\
    else\
        echo "CUDA_VERSION is not supported";\
        exit 1;\
    fi &&\
    honeybee_venv/bin/pip install -r requirements.txt &&\
    honeybee_venv/bin/pip install -r requirements_demo.txt &&\
    honeybee_venv/bin/pip install flask sentencepiece opencv-python

# download honeybee checkpoints
mkdir checkpoints &&\
    wget -q https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M144.tar.gz -O checkpoints/7B-C-Abs-M144.tar.gz &&\
    tar -xvzf checkpoints/7B-C-Abs-M144.tar.gz -C checkpoints &&\
    rm -f checkpoints/7B-C-Abs-M144.tar.gz
wget -q https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-D-Abs-M144.tar.gz -O checkpoints/7B-D-Abs-M144.tar.gz &&\
    tar -xvzf checkpoints/7B-D-Abs-M144.tar.gz -C checkpoints &&\
    rm -f checkpoints/7B-D-Abs-M144.tar.gz
wget -q https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M256.tar.gz -O checkpoints/7B-C-Abs-M256.tar.gz &&\
    tar -xvzf checkpoints/7B-C-Abs-M256.tar.gz -C checkpoints &&\
    rm -f checkpoints/7B-C-Abs-M256.tar.gz
