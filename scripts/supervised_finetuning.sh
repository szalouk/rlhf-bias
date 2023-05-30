#!/bin/bash

MODEL_PATH=$1

torchrun --nnodes 1  --nproc_per_node 4 \
	supervised_finetuning.py \
	--model_path=${MODEL_PATH} \
	--streaming \
	--no_gradient_checkpointing \
	--learning_rate 1e-5 \
	--max_steps 5000 \
	--batch_size 4 \
	--bf16 \
	--output_dir ./supervised_finetuning_out
