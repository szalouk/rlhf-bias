#!/bin/bash

MODEL_NAME=$1

#torchrun --nnodes 1  --nproc_per_node 1 \
python \
	examples/stack_llama/scripts/reward_modeling.py \
	--model_name=${MODEL_NAME} \
	--bf16=False

