#!/bin/bash

source /home/staff/d/danthes/setup_H100.sh
/home/staff/d/danthes/miniconda3/envs/avalanche-h100/bin/python train_models.py "$@"