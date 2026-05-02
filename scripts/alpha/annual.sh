# alpha=0: dot-product attention term only
# alpha=1: wedge/geometric term only

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-data/annual.csv}"
OUT_ROOT="${OUT_ROOT:-runs/alpha_annual}"
DEVICE="${DEVICE:-cpu}"   # cuda or cpu

ALPHAS=(0 0.25 0.5 0.75 1)
SEEDS=(0 1 7 42 65 67 1738 2001 2004 2025 2026 27182 31415 77777 99999)

# shared settings
LENGTH=6
PRED_LEN=3
PSEUDO_LEN=32
M=1
LAYERS=1
FF_DIM=32
DROPOUT=0.1
ATTN_DROPOUT=0.1
PAD_MODE="circular"

# annual-specific starter settings
# 0.6/0.2 split avoids empty val/test window loaders for tiny dataset
BATCH_SIZE=4
EPOCHS=10
LR=0.001
WEIGHT_DECAY=0.0
WV="db1"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "dataset,alpha,seed,test_loss,test_mse,test_mae" > "$SUMMARY"

for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="$OUT_ROOT/alpha${ALPHA//./p}/seed${SEED}"
    mkdir -p "$SAVE_DIR"

    echo "============================================================"
    echo "Annual FX | alpha=${ALPHA} | seed=${SEED}"
    echo "Saving to: ${SAVE_DIR}"
    echo "============================================================"

    "$PYTHON_BIN" custom_run.py \
      --mode train \
      --dataset_type annual \
      --data_path "$DATA_PATH" \
      --save_dir "$SAVE_DIR" \
      --variables 20 \
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
      --attention_type geometric

    METRICS="$(tail -n 1 "$SAVE_DIR/test_metrics.csv")"
    echo "annual,${ALPHA},${SEED},${METRICS}" >> "$SUMMARY"
  done
done

echo "Done. Summary written to: $SUMMARY"
