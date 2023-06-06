#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# conda init bash
# conda activate kcalibration

GPUS=1
LOG_FOLDER="./logs/atlas/"
echo "Log Folder: " ${LOG_FOLDER}
mkdir -p ${LOG_FOLDER}

MODEL_NAME=$1

JOBNAME="${MODEL_NAME}_eval"

WRAP="
python evaluate_bias.py --model_name=${MODEL_NAME}
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