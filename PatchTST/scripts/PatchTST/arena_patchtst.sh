#!/bin/bash

# Get absolute path to the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Go one level up to reach DLinear/
PROJECT_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_DIR"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/PatchTST/logs/LongForecasting"

# Set the path to the data files
DATA_DIR="$PROJECT_DIR/../main_dataset/count_data"

# Check that it exists and has files
if [ ! -d "$DATA_DIR" ]; then
  echo "Directory does not exist: $DATA_DIR"
  exit 1
fi

files=($DATA_DIR/*)
total_files=${#files[@]}
index=1

for data_file in $DATA_DIR/*.csv; do
	if [ -f "$data_file" ]; then
		filename=$(basename "$data_file")
		model_id="${filename%.*}"

		echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing $index/$total_files: $filename"

		python -u ../PatchTST/run_longExp.py \
		--random_seed 2025 \
		--is_training 1 \
		--root_path $DATA_DIR \
		--data_path $filename \
		--model_id $model_id \
		--model PatchTST \
		--data custom \
		--features MS \
		--target count \
		--seq_len 96 \
		--pred_len 96 \
		--enc_in 5 \
		--dec_in 5 \
		--e_layers 3 \
		--n_heads 4 \
		--d_model 16 \
		--d_ff 128 \
		--dropout 0.3 \
		--fc_dropout 0.3 \
		--head_dropout 0 \
		--patch_len 16 \
		--stride 1 \
		--des 'Exp' \
		--train_epochs 1 \
		--freq t \
		--itr 1 --batch_size 128 --learning_rate 0.0001 >../PatchTST/logs/LongForecasting/${model_id}.log
		((index++))
	fi

	if [ ! -f "$data_file" ]; then
		  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ Skipping: $data_file does not exist or is not a regular file." >&2
		  continue
	fi
done
