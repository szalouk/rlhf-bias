#!/bin/bash

MODEL_NAME=$1

torchrun --nnodes 1  --nproc_per_node 4 \
	examples/stack_llama/scripts/reward_modeling.py \
	--model_name=${MODEL_NAME} \
	--bf16=True \
	--per_device_train_batch_size=2 \
	--learning_rate=1e-5 \
	--eval_steps=500

