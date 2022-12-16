#!/bin/bash

DATA_NAME="/home/derekhmd/summ_bot/data/GodelInput.20221216/cods1"
OUTPUT_DIR="/home/derekhmd/summ_bot/output_cods1"
MODEL_PATH="microsoft/GODEL-v1_1-base-seq2seq"
EXP_NAME="20221216_derek_cods1"

curr_dir=$(pwd)
cd GODEL/GODEL

python train.py --model_name_or_path ${MODEL_PATH} \
	--dataset_name ${DATA_NAME} \
	--output_dir ${OUTPUT_DIR} \
	--per_device_train_batch_size=4 \
	--per_device_eval_batch_size=4 \
	--max_target_length 512 \
	--max_length 512 \
	--num_train_epochs 2 \
	--save_steps 10000 \
	--num_beams 5 \
	--exp_name ${EXP_NAME} --preprocessing_num_workers 4

cd ${curr_dir}
