#!/bin/bash

# This is a script to run some initial baseline experiments for
# Project milestone.

# Note this assumes that you've pre-downloaded the data for the different classes
# already.

# Run first experiment on {3, 10, 50} classes.

MAX_ITER=100
echo "Running baseline on 3 classes !"
python baselinev2.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_3/ --experiment_name="3_classes"

# Run experiment on 10 classes
echo "Running baseline on 10 classes !"
python baselinev2.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_10/ --experiment_name="10_classes"

# Run experiment on 50 classes
echo "Running baseline on 50 classes !"
python baselinev2.py --max_iter=$MAX_ITER --data_folder=data/numpy_bitmap_50/ --experiment_name="50_classes"

# All done
echo "All done !"
