#!/bin/bash

set -x
./run_simple_cnn_model_experiments_gpu.sh
./run_simple_cnn_model_with_binarized_data_experiments_gpu.sh
./run_simple_cnn_modelv2_experiments_gpu.sh
./run_simple_cnn_modelv2_with_binarized_data_experiments_gpu.sh
