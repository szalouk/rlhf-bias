#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# conda init bash
# conda activate kcalibration

GPUS=4
echo "Number of GPUs: " ${GPUS}
LOG_FOLDER="./logs/atlas/"
echo "Log Folder: " ${LOG_FOLDER}
mkdir -p ${LOG_FOLDER}

MODEL_NAME=$1
REWARD_MODEL_NAME=$2
NUM_TRAINING_EXAMPLES=$3
JOBNAME=${MODEL_NAME}

WRAP="
accelerate launch --multi_gpu --num_machines 1 --num_processes ${GPUS} \
	rl_training.py \
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
	--mini_batch_size=2 \
	--batch_size=16 \
	--eval_steps=100 \
	--num_training_examples=${NUM_TRAINING_EXAMPLES}
"
echo $WRAP

sbatch --output=${LOG_FOLDER}/%j.out \
       --error=${LOG_FOLDER}/%j.err \
       --nodes=1 \
       --ntasks-per-node=1 \
       --time=2-00:00:00 \
       --mem=64G \
       --partition=atlas \
       --account=atlas \
       --cpus-per-task=16 \
       --gres=gpu:a5000:${GPUS} \
       --job-name="${JOBNAME}" \
       --wrap="${WRAP}" \
       --dependency=singleton \
       --exclude=atlas6,atlas7,atlas8,atlas10