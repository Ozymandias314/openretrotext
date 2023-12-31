#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$TRAIN_FILE":/app/openretro/data/tmp_for_docker/raw_train.csv \
  -v "$PROCESSED_DATA_PATH_LOCALRETRO":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_LOCALRETRO":/app/openretro/checkpoints/tmp_for_docker \
  -t openretro:gpu \
  python train.py \
  --do_train \
  --model_name="localretro" \
  --data_name="$DATA_NAME" \
  --log_file="localretro_train_$DATA_NAME" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker
