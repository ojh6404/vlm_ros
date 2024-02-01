#!/bin/bash

if [[ "$1" == "llava" ]]; then
    cd LLaVA && llava_venv/bin/python server.py
elif [[ "$1" == "honeybee" ]]; then
    cd honeybee && honeybee_venv/bin/python server.py
else
    echo "Invalid argument. Please specify 'llava' or 'honeybee'."
fi
