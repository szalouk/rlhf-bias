# Measuring and Reducing Bias in LLMs introduced by RLHF
CS224R Final Project

## RLHF Training
There are three main steps to the RLHF training process:
1. Supervised fine-tuning of the base LLM to create SFT LLM:
    - `./scripts/supervised_finetuning.sh <SFT_MODEL_NAME>`
2. Reward modeling using dialog pairs from the StackExchange dataset using SFT LLM to create RM:
    - `scripts/reward_modeling.sh <RM_MODEL_NAME>`
3. RL fine-tuning of SFT LLM with the reward model:
    - `./scripts/rl_training.sh <SFT_MODEL_NAME> <RM_MODEL_NAME> <NUM_TRAINING_EXAMPLES>`

LoRA layers were using at all stages to reduce memory requirements. 
At each stage the peft adapter layers were merged with the base model, using: 
```shell
python merge_peft_adapter.py --adapter_model_name=XXX --base_model_name=YYY --output_name=ZZZ
```
Note that this script requires `peft>=0.3.0`.


## Evaluation

### Bias
To evaluate the bias of finetuned and debiased GPT-Neo models, run:
```shell
python self_debiasing.py --models <MODEL_1> <MODEL_2> ... --modes default debiased
```

For LLAMA models, run:
```shell
python self_debiasing_llama.py --models <MODEL_1> <MODEL_2> ... --modes default debiased
```

### Perplexity
To evaluate the perplexity of finetuned and debiased models, run:
```shell
python eval_perplexity.py --models <MODEL_1> <MODEL_2> ... --modes default debiased
```
