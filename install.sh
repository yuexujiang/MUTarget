#!/bin/bash

#conda create --name env_test python=3.10
#conda activate env_test
conda install pytorch==1.13 pytorch-cuda=11.7 torchvision=0.14.0 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup