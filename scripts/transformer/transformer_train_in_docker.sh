#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PROCESSED_DATA_PATH_TRANSFORMER":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_TRANSFORMER":/app/openretro/checkpoints/tmp_for_docker \
  -t openretro:gpu \
  python train.py \
  --do_train \
  --data="do_not_change_this" \
  --model_name="transformer" \
  --data_name="$DATA_NAME" \
  --log_file="transformer_train_$DATA_NAME" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  -seed 42 \
  -gpu_ranks 0 \
  -save_checkpoint_steps 10000 \
  -keep_checkpoint 10 \
  -train_steps 125000 \
  -param_init 0 \
  -param_init_glorot \
  -max_generator_batches 32 \
  -batch_size 32 \
  -batch_type sents \
  -normalization sents \
  -max_grad_norm 0 \
  -optim adam \
  -adam_beta1 0.9 \
  -adam_beta2 0.998 \
  -decay_method noam \
  -warmup_steps 8000 \
  -learning_rate 2 \
  -label_smoothing 0.0 \
  -report_every 1000 \
  -layers 4 \
  -rnn_size 256 \
  -word_vec_size 256 \
  -encoder_type transformer \
  -decoder_type transformer \
  -dropout 0.1 \
  -position_encoding \
  -share_embeddings \
  -global_attention general \
  -global_attention_function softmax \
  -self_attn_type scaled-dot \
  --heads 8 \
  -transformer_ff 2048