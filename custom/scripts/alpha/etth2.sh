# alpha=0: dot-product attention term only
# alpha=1: wedge/geometric term only

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-data/ETTh2.csv}"
OUT_ROOT="${OUT_ROOT:-runs/alpha_etth2}"
DEVICE="${DEVICE:-cpu}"   # cuda or cpu

ALPHAS=(0 0.25 0.5 0.75 1)
SEED=2025

# shared settings
LENGTH=96
PRED_LEN=96
PSEUDO_LEN=32
M=1
LAYERS=1
FF_DIM=32
DROPOUT=0.1
ATTN_DROPOUT=0.1
PAD_MODE="circular"

# ETTh2 SimpleTM baseline settings
BATCH_SIZE=256
EPOCHS=10
LR=0.006
WEIGHT_DECAY=0.0
WV="bior3.1"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "dataset,alpha,seed,test_loss,test_mse,test_mae" > "$SUMMARY"

for ALPHA in "${ALPHAS[@]}"; do
  SAVE_DIR="$OUT_ROOT/alpha${ALPHA//./p}"
  mkdir -p "$SAVE_DIR"

  echo "============================================================"
  echo "ETTh2 | alpha=${ALPHA} | seed=${SEED}"
  echo "Saving to: ${SAVE_DIR}"
  echo "============================================================"

  "$PYTHON_BIN" custom_run.py \
    --mode train \
    --dataset_type etth2 \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" \
    --variables 7 \
    --length "$LENGTH" \
    --prediction_length "$PRED_LEN" \
    --pseudo_length "$PSEUDO_LEN" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --m "$M" \
    --wv "$WV" \
    --pad_mode "$PAD_MODE" \
    --alpha "$ALPHA" \
    --attention_dropout "$ATTN_DROPOUT" \
    --dropout "$DROPOUT" \
    --transformer_layers "$LAYERS" \
    --feedforward_dim "$FF_DIM" \
    --normalize \
    --is_geometric

  METRICS="$(tail -n 1 "$SAVE_DIR/test_metrics.csv")"
  echo "etth2,${ALPHA},${SEED},${METRICS}" >> "$SUMMARY"
done

echo "Done. Summary written to: $SUMMARY"
