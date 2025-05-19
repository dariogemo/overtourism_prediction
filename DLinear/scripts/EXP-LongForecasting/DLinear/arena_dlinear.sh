#!/bin/sh

# Get absolute path to the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Go one level up to reach DLinear/
PROJECT_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_DIR"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/logs/LongForecasting"

# Set the path to the data files
DATA_DIR="$PROJECT_DIR/../../main_dataset/count_data"

# Run the script and save logs inside project_root/DLinear/logs/...
for data_file in "$DATA_DIR"/*; do
  if [ -f "$data_file" ]; then
    filename=$(basename "$data_file")
    model_id="${filename%.*}"  # remove extension

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing $index/$total_files: $filename"

    python -u "$PROJECT_DIR/run_longExp.py" \
      --is_training 1 \
      --root_path "$DATA_DIR" \
      --data_path "$filename" \
      --model_id "$model_id" \
      --model DLinear \
      --data custom \
      --features MS \
      --target count \
      --c_out 1 \
      --seq_len 96 \
      --pred_len 96 \
      --enc_in 5 \
      --dec_in 5 \
      --freq t \
      --des 'Exp' \
      --itr 1 \
      --batch_size 64 \
      --train_epochs 30 \
      --patience 7 \
      > "$PROJECT_DIR/logs/LongForecasting/${model_id}.log"
    ((index++))
  fi
done
