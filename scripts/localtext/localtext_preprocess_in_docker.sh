#!/bin/bash
echo $TRAIN_FILE

docker run \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PWD/data/USPTO_50k/raw/raw_train_nn.csv":/app/openretro/data/tmp_for_docker/raw_train_nn.csv \
  -v "$PWD/data/USPTO_50k/raw/raw_val_nn.csv":/app/openretro/data/tmp_for_docker/raw_val_nn.csv \
  -v "$PWD/data/USPTO_50k/raw/raw_test_nn.csv":/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  -v "$PWD/data/USPTO_50k/processed_localtext":/app/openretro/data/tmp_for_docker/processed \
  -t openretro:gpu \
  python preprocess.py \
  --model_name="localtext" \
  --data_name="USPTO_50k" \
  --log_file="localtext_preprocess_USPTO_50k" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train_nn.csv \
  --val_file=/app/openretro/data/tmp_for_docker/raw_val_nn.csv \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --num_cores="32"
