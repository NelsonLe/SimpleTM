export CUDA_VISIBLE_DEVICES=0
model_name=SimpleTM

# Annual FX forecasting: multivariate -> single target (Australia)
# Uses all 21 country series as input and predicts Australia only.

python -u run.py \
  --is_training 1 \
  --lradj TST \
  --patience 3 \
  --root_path ./dataset/custom/ \
  --data_path proc_annual.csv \
  --model_id annual_fx_australia \
  --model $model_name \
  --data custom \
  --features MS \
  --target Australia \
  --freq y \
  --seq_len 6 \
  --label_len 0 \
  --pred_len 3 \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.001 \
  --batch_size 8 \
  --train_epochs 10 \
  --fix_seed 2025 \
  --use_norm 1 \
  --embed fixed \
  --wv db1 \
  --m 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 1 \
  --des Exp \
  --itr 1 \
  --alpha 1 \
  --l1_weight 5e-05