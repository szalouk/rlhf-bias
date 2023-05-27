#!/bin/bash

MODEL_PATH=$1

torchrun --nnodes 1  --nproc_per_node 1 \
	examples/stack_llama/scripts/supervised_finetuning.py \
	--model_path=${MODEL_PATH} \
	--streaming \
	--no_gradient_checkpointing \
	--learning_rate 1e-5 \
	--max_steps 5000 \
	--output_dir ./supervised_finetuning_out
