SEEDS=(7 42 67 1738 2025)

for SEED in ${SEEDS[@]}; do

  echo Using seed "$SEED"...

  # ETTh2
  python custom_run.py \
    --mode train \
    --dataset_type ett \
    --data_path data/ETTh2.csv \
    --save_dir runs/baseline/etth2_inverted_"$SEED" \
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
    --learnable_wavelets \
    --inverted \
    --seed "$SEED"

  # ETTh1
  python custom_run.py \
    --mode train \
    --dataset_type ett \
    --data_path data/ETTh1.csv \
    --save_dir runs/baseline/etth1_inverted_"$SEED" \
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
    --learnable_wavelets \
    --inverted \
    --seed "$SEED"

done