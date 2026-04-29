#!/bin/bash
ENV="./customtm_env"

SEEDS=(0 1 7 42 65 67 1738 2001 2004 2025 2026 27182 31415 77777 99999)

for SEED in ${SEEDS[@]}; do

  echo Using seed "$SEED"...

  # ETTh2
  "$ENV"/bin/python custom_run.py \
    --mode train \
    --dataset_type ett \
    --data_path data/ETTh2.csv \
    --save_dir runs/wavelets/etth2_fixed_"$SEED" \
    --variables 7 \
    --length 96 \
    --prediction_length 96 \
    --pseudo_length 32 \
    --batch_size 256 \
    --epochs 10 \
    --learning_rate 0.006 \
    --m 3 \
    --wv bior3.1 \
    --pad_mode circular \
    --alpha 0.1 \
    --attention_dropout 0.5 \
    --dropout 0.1 \
    --transformer_layers 1 \
    --feedforward_dim 32 \
    --normalize \
    --attention_type geometric \
    --seed "$SEED"

  # ETTh1
  "$ENV"/bin/python custom_run.py \
    --mode train \
    --dataset_type ett \
    --data_path data/ETTh1.csv \
    --save_dir runs/wavelets/etth1_fixed_"$SEED" \
    --variables 7 \
    --length 96 \
    --prediction_length 96 \
    --pseudo_length 32 \
    --batch_size 256 \
    --epochs 10 \
    --learning_rate 0.02 \
    --m 3 \
    --wv db1 \
    --pad_mode circular \
    --alpha 0.3 \
    --attention_dropout 0.5 \
    --dropout 0.1 \
    --transformer_layers 1 \
    --feedforward_dim 32 \
    --normalize \
    --attention_type geometric \
    --seed "$SEED"

done