#!/bin/bash

FLAGS=(--mem=8G --time=01:00:00 --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=1)

FILES=(scripts/*/*.sh)
for FILE in ${FILES[@]}; do
    BN=$(basename "$FILE" .sh)
    sbatch "${FLAGS[@]}" --output=logs/"$BN".out "$FILE"
    sleep 1
done