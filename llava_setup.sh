#!/usr/bin/env sh

# virtualenv for llava
git clone https://github.com/haotian-liu/LLaVA.git &&\
    cd LLaVA &&\
    python -m venv llava_venv &&\
    llava_venv/bin/pip install --upgrade pip setuptools wheel &&\
    llava_venv/bin/pip install -e . &&\
    llava_venv/bin/pip install flask opencv-python protobuf
