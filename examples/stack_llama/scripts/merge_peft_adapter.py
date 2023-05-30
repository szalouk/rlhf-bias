from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel
from peft.utils import _get_submodules
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "LoRA model checkpoint"})
    base_model_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to model checkpoint"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "Base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "Output model name"})
    is_reward_model: Optional[bool] = field(default=False, metadata={"help": "Set to True if the base-model is a reward model."},)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_checkpoint is not None, "please provide the checkpoint to the base model"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"


if script_args.is_reward_model:
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_checkpoint, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_checkpoint, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)
config = AutoConfig.from_pretrained(script_args.base_model_name)
architecture = config.architectures[0]
if "Llama" in architecture:
    print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )

# Load the Lora model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

# key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
# for key in key_list:
#     parent, target, target_name = _get_submodules(model.base_model.model, key)
#     if isinstance(target, peft.tuners.lora.Linear):
#         bias = target.bias is not None
#         new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
#         model.base_model._replace_module(parent, target_name, new_module, target)

# model = model.base_model.model

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)
