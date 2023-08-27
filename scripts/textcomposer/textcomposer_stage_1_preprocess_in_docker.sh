#!/bin/bash

docker run --gpus \"device=4\"  \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PWD/data/USPTO_50k/raw/raw_train_nn.csv":/app/openretro/data/tmp_for_docker/raw_train_nn.csv \
  -v "$PWD/data/USPTO_50k/raw/raw_val_nn.csv":/app/openretro/data/tmp_for_docker/raw_val_nn.csv \
  -v "$PWD/data/USPTO_50k/raw/raw_test_nn.csv":/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  -v "$PWD/data/USPTO_50k/processed_textcomposer":/app/openretro/data/tmp_for_docker/processed \
  -v "$PWD/retro_b512_ep400":/app/openretro/retro_b512_ep400 \
  -v "$PWD/USPTO_rxn_corpus.csv":/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.csv \
  -v "$PWD/USPTO_rxn_corpus.pkl":/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.pkl \
  -t openretro:gpu \
  python preprocess.py \
  --model_name="textcomposer" \
  --stage=1 \
  --data_name="USPTO_50k" \
  --log_file="textcomposer_preprocess_s1_USPTO_50k" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train_nn.csv \
  --val_file=/app/openretro/data/tmp_for_docker/raw_val_nn.csv \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test_nn.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --prod_k=1 \
  --react_k=1 \
  --num_cores="32" \
  --encoder_path=allenai/scibert_scivocab_uncased \
  --corpus_file=/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.csv \
  --cache_path=/app/openretro/data/tmp_for_docker/ \
  --nn_path=/app/openretro/retro_b512_ep400 \
  --num_neighbors 3\
  --use_gold_neighbor \
  --max_length 512\
