#!/bin/sh

# Get absolute path to the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Go one level up to reach DLinear/
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_DIR"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/Informer2020/logs"

# Set the path to the data files
DATA_DIR="$PROJECT_DIR/../main_dataset/count_data"

# Check that it exists and has files
if [ ! -d "$DATA_DIR" ]; then
  echo "Directory does not exist: $DATA_DIR"
  exit 1
fi

# files=($DATA_DIR/*)
# total_files=${#files[@]}
data_file=$DATA_DIR/data_casa_di_giulietta.csv

if [ -f "$data_file" ]; then
	filename=$(basename "$data_file")
	model_id="${filename%.*}"

	echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing $filename"

	python -u ../Informer2020/main_informer.py \
	  --model informer \
	  --root_path $DATA_DIR \
	  --data_path $filename \
	  --data custom \
	  --features MS \
	  --target count \
	  --freq t \
	  --seq_len 96 \
	  --label_len 48 \
	  --pred_len 96 \
	  --enc_in 5 \
	  --dec_in 5 \
	  --c_out 1 \
	  --train_epochs 10 \
	  --patience 3 \
	  --batch_size 32 \
	  --e_layers 2 \
	  --d_layers 1 \
	  --attn prob \
	  --des 'Exp' \
	  --itr 1 \
	> "$PROJECT_DIR/Informer2020/logs/${model_id}.log"
fi

if [ ! -f "$data_file" ]; then
	  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ Skipping: $data_file does not exist or is not a regular file." >&2
fi
