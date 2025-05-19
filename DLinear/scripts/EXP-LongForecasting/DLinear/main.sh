#!/bin/bash

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./../../main_dataset/count_data/ \
  --data_path data_anfiteatro_arena.csv \
  --model_id arena_full_96 \
  --model DLinear \
  --data custom \
  --features MS \
  --target count \
  --c_out 1 \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 4 \
  --freq t \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  >logs/LongForecasting/DLinear_arena_96_96.log
