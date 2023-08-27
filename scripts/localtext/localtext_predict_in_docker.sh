#!/bin/bash

docker run --gpus \"device=7\" \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints_mapping_1e-4":/app/openretro/checkpoints \
  -v "$PWD/results_mapping_1e-4":/app/openretro/results \
  -v "$PWD/data/USPTO_50k/raw/raw_train_nn.csv":/app/openretro/data/tmp_for_docker/raw_train_nn.csv \
  -v "$PWD/data/USPTO_50k/raw/raw_val_nn.csv":/app/openretro/data/tmp_for_docker/raw_val_nn.csv \
  -v "$PWD/data/USPTO_50k/raw/raw_test_nn.csv":/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  -v "$PWD/data/USPTO_50k/processed_localtext":/app/openretro/data/tmp_for_docker/processed \
  -v "$PWD/checkpoints/USPTO_50k_localtext":/app/openretro/checkpoints/tmp_for_docker \
  -v "$PWD/results_mapping_1e-4/USPTO_50k_localtext":/app/openretro/results/tmp_for_docker \
  -v "$PWD/USPTO_rxn_corpus.csv":/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.csv \
  -v "$PWD/USPTO_rxn_corpus.pkl":/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.pkl \
  -v "$PWD/retro_b512_ep400":/app/openretro/retro_b512_ep400 \
  -t openretro:gpu \
  python predict.py \
  --model_name="localtext" \
  --data_name="USPTO_50k" \
  --log_file="localtext_predict_USPTO_50k" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train_nn.csv \
  --val_file=/app/openretro/data/tmp_for_docker/raw_val_nn.csv \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --test_output_path=/app/openretro/results/tmp_for_docker \
  --encoder_path=allenai/scibert_scivocab_uncased \
  --corpus_file=/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.csv \
  --cache_path=/app/openretro/data/tmp_for_docker/ \
  --nn_path=/app/openretro/retro_b512_ep400 \
  --num_neighbors 3\
  --use_gold_neighbor \
  --max_length 512\
