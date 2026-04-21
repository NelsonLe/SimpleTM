export CUDA_VISIBLE_DEVICES=0

python custom_run.py \
  --mode train \
  --dataset_type etth2 \
  --data_path data/ETTh2.csv \
  --save_dir runs/etth2_geom \
  --variables 7 \
  --length 96 \
  --prediction_length 96 \
  --pseudo_length 32 \
  --batch_size 32 \
  --epochs 3 \
  --learning_rate 0.001 \
  --m 2 \
  --wv db1 \
  --pad_mode circular \
  --alpha 1.0 \
  --attention_dropout 0.1 \
  --dropout 0.8 \
  --transformer_layers 1 \
  --feedforward_dim 32 \
  --normalize \
  --is_geometric

python custom_run.py \
  --mode train \
  --dataset_type annual \
  --data_path data/annual.csv \
  --save_dir runs/annual_geom \
  --variables 21 \
  --length 3 \
  --prediction_length 1 \
  --pseudo_length 1 \
  --batch_size 2 \
  --epochs 3 \
  --learning_rate 0.001 \
  --m 1 \
  --wv db1 \
  --pad_mode circular \
  --alpha 0.5 \
  --attention_dropout 0.1 \
  --dropout 0.8 \
  --transformer_layers 1 \
  --feedforward_dim 32 \
  --normalize \
  --is_geometric