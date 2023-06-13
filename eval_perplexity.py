import argparse
import torch
from tqdm import tqdm
import random

from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

from self_debiasing.modeling import GPT2Wrapper, LlamaWrapper

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

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
    parser.add_argument("--models", type=str, nargs='+', default=['EleutherAI/gpt-neo-125m'],
                        help="The specific models to run self-debiasing experiments for (e.g., 'gpt2-medium gpt2-large')")
    parser.add_argument("--modes", nargs='+', choices=['default', 'debiased'], default=['default', 'debiased'],
                        help="Whether to perform debiased ('debiased') or regular ('default') generation")
    parser.add_argument("--decay_constant", type=float, default=50,
                        help="Value for the decay constant (lambda in the paper)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    parser.add_argument("--max_length", type=int, default=-1,
                        help="The maximum input length to be processed (-1 corresponds to the model's context window)")
    parser.add_argument("--max_length_pattern", type=int, default=32,
                        help="The number of tokens to reserve for the self-diagnosis patterns")
    parser.add_argument("--stride", type=int, default=-1,
                        help="If set, for the first --stride tokens no loss is computed")
    parser.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    parser.add_argument("--no_cuda", action='store_true',
                        help="If set to true, all computations are done on CPU")
    parser.add_argument("--debug", action='store_true',
                        help="If set, additional debugging output is printed to stdout")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for initializing the random number generator used for sampling")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # num_training_examples = 1000
    # eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
    # eval_dataset = eval_dataset.select(range(num_training_examples))

    # original_columns = eval_dataset.column_names
    # num_proc = 24

    # def preprocess_function(examples):
    #     new_examples = {
    #         "text": []
    #     }
    #     for question, response_j in zip(examples["question"], examples["response_j"]):
    #         text = "Question: " + question + "\n\nAnswer: " + response_j
    #         new_examples["text"].append(text)

    #     return new_examples

    # eval_dataset = eval_dataset.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=num_proc,
    #     remove_columns=original_columns,
    # )
    eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    for model_name in args.models:
        print(f'Name: {model_name}')

        if 'gpt' in model_name.lower():
            wrapper = GPT2Wrapper(model_name=model_name, use_cuda=not args.no_cuda)
        elif 'llama' in model_name.lower():
            wrapper = LlamaWrapper(model_name=model_name, use_cuda=not args.no_cuda)
        else:
            assert False, f"Model name {model_name} not supported."
        
        wrapper._model.eval()
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = wrapper._tokenizer
        device = 'cuda' if not args.no_cuda else 'cpu'

        encodings = tokenizer('\n\n'.join(eval_dataset['text']), return_tensors='pt')

        max_length = (args.max_length if args.max_length > 0 else wrapper._model.config.max_position_embeddings) - args.max_length_pattern

        if args.stride <= 0:
            args.stride = max_length

        lls_debiased, lls_regular = [], []
        ppl_debiased, ppl_regular = None, None

        for i in tqdm(range(0, encodings.input_ids.size(1), args.stride)):
            begin_loc = max(i + args.stride - max_length, 0)
            end_loc = min(i + args.stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            debiasing_prefixes = DEBIASING_PREFIXES if not args.use_keywords else DEBIASING_KEYWORDS

            with torch.no_grad():
                loss_regular = wrapper.compute_loss(input_ids, labels=target_ids)
                log_likelihood_regular = loss_regular * trg_len

                if 'debiased' in args.modes:
                    loss_debiased = wrapper.compute_loss_self_debiasing(input_ids=input_ids, trg_len=trg_len, debiasing_prefixes=debiasing_prefixes,
                                                                        decay_constant=args.decay_constant, epsilon=args.epsilon, debug=args.debug)
                    log_likelihood_debiased = loss_debiased * trg_len

            lls_regular.append(log_likelihood_regular)
            ppl_regular = torch.exp(torch.stack(lls_regular).sum() / end_loc)

            if 'debiased' in args.modes:
                lls_debiased.append(log_likelihood_debiased)
                ppl_debiased = torch.exp(torch.stack(lls_debiased).sum() / end_loc)
                # print(f'Perplexity after {i} tokens: {ppl_debiased} (debiased) vs {ppl_regular} (regular)')
            # else:
            #     print(f'Perplexity after {i} tokens: {ppl_regular} (regular)')

        if 'debiased' in args.modes:
            print(f'Final perplexity: {ppl_debiased} (debiased) vs {ppl_regular} (regular)')
        else:
            print(f'Final perplexity: {ppl_regular} (regular)')

        output_name = f'{model_name.split("/")[-1]}_perplexity.txt'
        with open(output_name, 'a', encoding='utf8') as fh:
            fh.write(f'=== RESULT [{model_name}] ===\n')
            fh.write(f'Perplexity (regular):  {ppl_regular}\n')
            if 'debiased' in args.modes:
                fh.write(f'Perplexity (debiased): {ppl_debiased}\n\n')
