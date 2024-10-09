#!/usr/bin/env bash


# openslide via conda
conda install -c conda-forge openslide

# torch and torchvision with cuda
pip install --root-user-action=ignore torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# regular pip requirements
pip install --root-user-action=ignore -r pip_requirements.txt

