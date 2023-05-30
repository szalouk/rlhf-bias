#!/bin/bash

MODEL_NAME=$1
REWARD_MODEL_NAME=$2
NUM_TRAINING_EXAMPLES=$3

#python3 \
accelerate launch --num_machines 1 --num_processes 1 \
	examples/stack_llama/scripts/rl_training.py \
	--log_with=wandb \
	--model_name=${MODEL_NAME} \
	--reward_model_name=${REWARD_MODEL_NAME} \
	--adafactor=False \
	--tokenizer_name=${MODEL_NAME} \
	--save_freq=100 \
	--output_max_length=128 \
	--gradient_accumulation_steps=8 \
	--batched_gen=True \
	--ppo_epochs=4 \
	--seed=0 \
	--learning_rate=1.4e-5 \
	--early_stopping=True \
	--output_dir=${MODEL_NAME}-rl-finetune \
	--mini_batch_size=1 \
	--batch_size=16 \
	--eval_steps=1 \
	--num_training_examples=${NUM_TRAINING_EXAMPLES}

