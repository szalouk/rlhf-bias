# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from metrics import *

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

# tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.

    # TODO: See if comment above actually matters?
    model_name: Optional[str] = field(
        default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(
        default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(
        default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(
        default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(  # TODO: Enable?
        default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(  # TODO: Enable?
        default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(  # TODO: Change default/pass in higher target KL
        default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(
        default=None, metadata={"help": "n steps to save the model"})  # Cmd param = 100
    output_dir: Optional[str] = field(
        default="runs/", metadata={"help": "directory to save model(s)"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    dataset: Optional[str] = field(
        default="lvwerra/stack-exchange-paired",
        metadata={"help": "Dataset (as defined by HuggingFace library)"}
    )
    num_training_examples: Optional[int] = field(default=10000, metadata={
                                                 "help": "Number of examples to truncate train dataset to"})
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "max length of 'Question:  + question + \\n\\nAnswer:' prompt"})
    eval_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Num steps before eval"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

train_dataset = load_dataset(
    script_args.dataset, data_dir="data/rl", split="train")
print(f'train_dataset initial size = {len(train_dataset)}')
train_dataset = train_dataset.select(range(script_args.num_training_examples))
print(f'train_dataset truncated size = {len(train_dataset)}')

bias_metrics = {}
# bias_metrics['toxicity'] = ToxicityMetric(num_examples=100)
# bias_metrics['bold'] = BoldMetric(num_examples=50)
# bias_metrics['winobias'] = WinoBiasMetric()
bias_metrics['honest'] = HonestMetric(num_examples=50)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True,
               "function_to_apply": "none", "batch_size": 16, "truncation": True}

tokenizer = AutoTokenizer.from_pretrained(
    script_args.tokenizer_name, use_auth_token=True)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if "llama" in script_args.tokenizer_name:
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset ().
def build_dataset(
    tokenizer, dataset_name=script_args.dataset, max_prompt_length=None
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) <
                   max_prompt_length, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(
    tokenizer, max_prompt_length=script_args.max_prompt_length)


def collator(data):
    # list of dictionaires --> dictionary of lists
    # data = list of dictionaries
    # "for key in data[0]" = get keys associated with every dict
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed (random, numpy, torch) before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    # device_map="auto", #TODO: Change arg?
    peft_config=lora_config,
)

# Required to finetune GPT2.
# if script_args.model_name.split("/")[-1].lower() == "gpt2":
model.config.pad_token_id = tokenizer.eos_token_id

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    # device_map="auto", #TODO: Change arg? Be consistent w/ model definition above
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
)

# Required to finetune GPT2.
# if script_args.reward_model_name.split("/")[-1].lower() == "gpt2":
sentiment_pipe.model.config.pad_token_id = tokenizer.eos_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,  # Ignored.
    "top_p": 1.0,  # Ignored.
    "do_sample": True,  # If False, does greedy sampling.
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length

# Samples random value between 32 and script_args.output_max_length = 128.
# generation_kwargs["max_new_tokens"] is set to this value = length of output.
# generation_kwargs["max_length"] = length of input + output; overridden by `max_new_tokens`.
output_length_sampler = LengthSampler(output_min_length, output_max_length)

print(f'ppo_trainer.dataloader len = {len(ppo_trainer.dataloader)}')

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(
        response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(
        output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step -- ValueHead used here
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

    if epoch % script_args.eval_steps == 0:
        bias_stats = {}
        # Compute bias/toxicity/fairness metrics
        for _, metric in bias_metrics.items():
            bias_stat = metric.compute(ppo_trainer.model, tokenizer, generation_kwargs)
            bias_stats.update(bias_stat)

        
        print(f'Bias stats = {bias_stats}')
        
        # Add them to stats dict
        stats.update(bias_stats)

    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
