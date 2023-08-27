#!/bin/bash

docker run --gpus \"device=4\"  \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PWD/data/USPTO_50k/processed_textcomposer":/app/openretro/data/tmp_for_docker/processed \
  -v "$PWD/checkpoints/USPTO_50k_textcomposer":/app/openretro/checkpoints/tmp_for_docker \
  -v "$PWD/retro_b512_ep400":/app/openretro/retro_b512_ep400 \
  -v "$PWD/USPTO_rxn_corpus.csv":/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.csv \
  -v "$PWD/USPTO_rxn_corpus.pkl":/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.pkl \
  -t openretro:gpu \
  python train.py \
  --do_train \
  --model_name="textcomposer" \
  --stage=1 \
  --data_name="USPTO_50k" \
  --log_file="textcomposer_train_s1_USPTO_50k" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --batch_size=32 \
  --epochs=80 \
  --encoder_path=allenai/scibert_scivocab_uncased \
  --corpus_file=/app/openretro/data/tmp_for_docker/USPTO_rxn_corpus.csv \
  --cache_path=/app/openretro/data/tmp_for_docker/ \
  --nn_path=/app/openretro/retro_b512_ep400 \
  --num_neighbors 3\
  --use_gold_neighbor \
  --max_length 512\
