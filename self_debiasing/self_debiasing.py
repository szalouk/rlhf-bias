import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict

import torch
from tqdm import tqdm

from datasets import load_dataset
import evaluate

from modeling import GPT2Wrapper
from io_utils import Prompt

DEBIASING_PREFIXES = [
    "The following text contains rude, disrespectful, or unreasonable language:\n",
    "The following text contains very hateful, aggressive, disrespectful language:\n",
    "The following text contains sexually explicit language:\n",
    "The following text contains a threat:\n",
    "The following text contains negative or hateful language targeting someone because of their identity:\n",
    "The following text contains swear words, curse words, or other obscene or profane language:\n"
]

DEBIASING_KEYWORDS = [
    "(rude) ", "(sexually explicit) ", "(hateful) ", "(aggressive) ", "(racist) ", "(threat) ", "(violent) ", "(sexist) "
]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs='+', choices=['default', 'debiased'], default=['default', 'debiased'],
                        help="Whether to perform debiased ('debiased') or regular ('default') generation")
    parser.add_argument("--decay_constant", type=float, default=50,
                        help="Value for the decay constant (lambda in the paper)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    parser.add_argument("--do_sample", action='store_true',
                        help="Whether to perform sampling instead of greedy decoding")
    parser.add_argument("--min_length", type=int, default=20,
                        help="Minimum length for the generated text")
    parser.add_argument("--max_length", type=int, default=20,
                        help="Maximum length for the generated text")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Only for sampling. If set, only the top_k tokens with the highest probabilities are considered.")
    parser.add_argument("--num_beams", type=int, default=3,
                        help="The number of beams for beam search")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="The number of sequences to return for each prompt")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="The number of sequences to return for each prompt. Increasing this has the same effect as increasing "
                             "--num_return_sequences, but the output sequences are generated sequentially instead of generating them in "
                             "parallel. This is slower but can prevent OOM errors for large models.")
    parser.add_argument("--not_challenging_only", action='store_true',
                        help="If set to true, all prompts are considered (as opposed to considering only 'challenging' promts)")
    parser.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for initializing the random number generator used for sampling")
    parser.add_argument("--debug", action='store_true',
                        help="If set, additional debugging output is printed to stdout")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_examples = 1000
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

    sampled_idx = random.sample(range(len(dataset)), num_examples)
    toxic_samples = dataset[sampled_idx]
    prompts = [p['text'] for p in toxic_samples['prompt']]

    model_name = "EleutherAI/gpt-neo-125m"
    wrapper = GPT2Wrapper(model_name=model_name)

    print(f'Generating continuations for {len(prompts)} prompts')

    toxicity = evaluate.load("toxicity")
    for mode in args.modes:
        debiasing_prefixes = (DEBIASING_PREFIXES if not args.use_keywords else DEBIASING_KEYWORDS) if mode == 'debiased' else []
        output_texts = []
        prompt_iterator = tqdm(prompts, desc="Prompts")
        for prompt in prompt_iterator:
            for _ in range(args.num_repeats):
                outputs = wrapper.generate_self_debiasing(
                    [prompt], debiasing_prefixes=debiasing_prefixes, decay_constant=args.decay_constant, epsilon=args.epsilon,
                    debug=args.debug, min_length=args.min_length, max_length=args.max_length, do_sample=args.do_sample,
                    num_beams=args.num_beams, top_k=args.top_k, num_return_sequences=args.num_return_sequences
                )
                # print("prompt:", prompt)
                # print("outputs:", outputs[0])
                output_texts.append(outputs[0])

        # print(output_texts)

        toxicity_ratio = toxicity.compute(predictions=output_texts, aggregation="ratio")
        max_toxicity = toxicity.compute(predictions=output_texts, aggregation="maximum")
        
        print(f'=== RESULT [{model_name}, {mode}] ===')
        print("max_toxicity:", max_toxicity)
        print("toxicity_ratio:", toxicity_ratio)
