#!/bin/bash

DATA_NAME="/home/derekhmd/summ_bot/data/DailySummaryCods1.20221215/train"
OUTPUT_DIR="/home/derekhmd/summ_bot/output"
MODEL_PATH="microsoft/GODEL-v1_1-base-seq2seq"
EXP_NAME="20221215_derek_cods1"

curr_dir=$(pwd)
cd GODEL/GODEL

python train.py --model_name_or_path ${MODEL_PATH} \
	--dataset_name ${DATA_NAME} \
	--output_dir ${OUTPUT_DIR} \
	--per_device_train_batch_size=16 \
	--per_device_eval_batch_size=16 \
	--max_target_length 512 \
	--max_length 512 \
	--num_train_epochs 50 \
	--save_steps 10000 \
	--num_beams 5 \
	--exp_name ${EXP_NAME} --preprocessing_num_workers 24

cd ${curr_dir}
