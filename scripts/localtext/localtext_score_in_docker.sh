#!/bin/bash

docker run \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/data/USPTO_50k/raw/raw_test_nn.csv":/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  -v "$PWD/results_mapping_1e-4/USPTO_50k_localtext":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python score.py \
  --model_name="localtext" \
  --log_file="localtext_score_USPTO_50k" \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  --prediction_file=/app/openretro/results/tmp_for_docker/predictions.csv
