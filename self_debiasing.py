import argparse
import json
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
import evaluate

import random
import json
from tqdm import tqdm

from self_debiasing.modeling import GPT2Wrapper

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

def get_random_prompts(dataset, num_examples=100):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = random.sample(range(len(dataset)), num_examples)
    return(dataset[picks])

class ToxicityMetric:
    def __init__(self, num_examples=100):
        toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")
        toxic_sample= get_random_prompts(toxicity_prompts, num_examples)
        self.toxic_prompts = [p['text'] for p in toxic_sample['prompt']]
        self.toxicity = evaluate.load("toxicity")
    
    def compute(self, model, tokenizer, debiasing_prefixes, args, max_length=50):
        model_continuations=[]

        for prompt in tqdm(self.toxic_prompts, desc='Evaluating Toxicity'):
            with torch.no_grad():
                outputs = model.generate_self_debiasing(
                    [prompt], debiasing_prefixes=debiasing_prefixes,
                    decay_constant=args.decay_constant, epsilon=args.epsilon,
                    max_length=max_length, do_sample=False, pad_token_id=50256)
            continuation = outputs[0]
            model_continuations.append(continuation)
        
        print(list(zip(self.toxic_prompts[:10], model_continuations[:10])))

        toxicity_ratio = self.toxicity.compute(predictions=model_continuations, aggregation="ratio")
        max_toxicity = self.toxicity.compute(predictions=model_continuations, aggregation="maximum")

        metrics = {
            'bias/toxicity/max_toxicity' : max_toxicity['max_toxicity'],
            'bias/toxicity/toxicity_ratio' : toxicity_ratio['toxicity_ratio']
        }
        return metrics

class BoldMetric:
    def __init__(self):
        bold = load_dataset("AlexaAI/bold", split="train")
        female_bold = [p for p in bold if p['category'] == 'American_actresses']
        male_bold = [p for p in bold if p['category'] == 'American_actors']
        
        num_prompts = min(len(male_bold), len(female_bold))
        print(f'Num prompts = {num_prompts}')

        self.prompts = {}
        self.prompts['male'] = [p['prompts'][0] for p in male_bold][:num_prompts]
        self.prompts['female'] = [p['prompts'][0] for p in female_bold][:num_prompts]
        self.regard = evaluate.load('regard', 'compare')
    
    def compute(self, model, tokenizer, debiasing_prefixes, args, max_length=50):
        continuations = {k: [] for k in ['male', 'female']}

        for gender, prompts in self.prompts.items():
            for prompt in tqdm(prompts, desc=f'Evaluating Bold for {gender}'):
                outputs = model.generate_self_debiasing(
                    [prompt], debiasing_prefixes=debiasing_prefixes,
                    decay_constant=args.decay_constant, epsilon=args.epsilon,
                    max_length=max_length, do_sample=False, pad_token_id=50256)
                continuation = outputs[0]
                continuations[gender].append(continuation)
        
        regard = self.regard.compute(data=continuations['male'], references=continuations['female'])

        metrics = {
            'bias/bold/regard_difference/positive' : regard['regard_difference']['positive'],
            'bias/bold/regard_difference/neutral' : regard['regard_difference']['neutral'],
            'bias/bold/regard_difference/other' : regard['regard_difference']['other'],
            'bias/bold/regard_difference/negative' : regard['regard_difference']['negative']
        }
        return metrics

class HonestMetric:
    def __init__(self):
        honest_dataset_queer = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
        honest_dataset_binary = load_dataset("MilaNLProc/honest", 'en_binary', split='honest')
        
        self.prompts = {}
        self.prompts['queer'] = [h['template_masked'].replace(' [M].','') for h in honest_dataset_queer if h['category'].startswith('queer')]
        self.prompts['nonqueer'] = [h['template_masked'].replace(' [M].','') for h in honest_dataset_queer if h['category'].startswith('nonqueer')]
        self.prompts['male'] = [h['template_masked'].replace(' [M].','') for h in honest_dataset_binary if h['category'].startswith('male')]
        self.prompts['female'] = [h['template_masked'].replace(' [M].','') for h in honest_dataset_binary if h['category'].startswith('female')]
        
        self.generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,  # Ignored.
            "top_p": 1.0,  # Ignored.
            "do_sample": True,  # If False, does greedy sampling.
            "eos_token_id": 100_000,
        }
        self.honest = evaluate.load('honest', 'en')
    
    def compute(self, model, tokenizer, debiasing_prefixes, args, num_generations=20):
        continuations = {k: [] for k in ['queer', 'nonqueer', 'male', 'female']}

        for gender, prompts in self.prompts.items():
            for prompt in tqdm(prompts, desc=f'Evaluating honest for {gender}'):
                inputs = tokenizer(prompt, return_tensors="pt")
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                max_len = inputs["input_ids"].shape[-1] + 10
                outputs = model.generate_self_debiasing(
                    [prompt], debiasing_prefixes=debiasing_prefixes,
                    decay_constant=args.decay_constant, epsilon=args.epsilon,
                    pad_token_id=tokenizer.pad_token_id, max_length=max_len,
                    num_return_sequences=num_generations, **self.generation_kwargs)
                continuation = outputs[0]
                continuations[gender].append(continuation)

        groups = ['queer'] * 50 + ['nonqueer'] * 50 + ['male'] * 50 + ['female'] * 50
        continuations = [c.split() for c in continuations['queer']] + [q.split() for q in continuations['nonqueer']] + [c.split() for c in continuations['male']] + [q.split() for q in continuations['female']]

        honest_score = self.honest.compute(predictions=continuations, groups=groups)
        print(f'honest_score = {honest_score}')
        metrics = {
            'bias/honest/queer' : honest_score['honest_score_per_group']['queer'],
            'bias/honest/nonqueer' : honest_score['honest_score_per_group']['nonqueer'],
            'bias/honest/male' : honest_score['honest_score_per_group']['male'],
            'bias/honest/female' : honest_score['honest_score_per_group']['female']
        }
        return metrics

class WinoBiasMetric:
    def __init__(self):
        data = load_dataset("mathemakitten/winobias_antistereotype_test", split="test")
        
        self.prompts1 = [data[i]['text'] + data[i]["classes"][0] for i in range(len(data))]
        self.prompts2 = [data[i]['text'] + data[i]["classes"][1] for i in range(len(data))]
        
        self.prompts = {}
        self.prompts['male'] = [data[i]['text'] + ' he ' for i in range(len(data))]
        self.prompts['female'] = [data[i]['text'] + ' she ' for i in range(len(data))]

        self.accuracy = evaluate.load("accuracy")
        self.toxicity = evaluate.load("toxicity")
    
    def eval_loss(self, prompt, model, tokenizer, debiasing_prefixes, args, device):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        loss = model.compute_loss_self_debiasing(input_ids, input_ids.size(1),
                                    debiasing_prefixes=debiasing_prefixes,
                                    decay_constant=args.decay_constant, epsilon=args.epsilon)
        loss = loss.detach().cpu().item()
        return loss
    
    def log_probs_from_logits(self, logits, labels):
        logp = F.log_softmax(logits, dim=-1)
        logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logp_label
    
    def sequence_logprob(self, model, labels, input_len=0):
        with torch.no_grad():
            output = model(labels)
            log_probs = self.log_probs_from_logits(output.logits[:,:-1, :], labels[:, 1:])
            seq_log_prob = torch.sum(log_probs[:, input_len:])
        return seq_log_prob.cpu().item()

    def compute(self, model, tokenizer, debiasing_prefixes, args):
        device = model._model.device

        preds = []
        for prompt1, prompt2 in tqdm(zip(self.prompts1, self.prompts2), desc='Evaluating WinoBias'):
            loss1 = self.eval_loss(prompt1, model, tokenizer, debiasing_prefixes, args, device)
            loss2 = self.eval_loss(prompt2, model, tokenizer, debiasing_prefixes, args, device)
            pred = 0 if loss1 < loss2 else 1
            preds.append(pred)

        labels = [0] * len(preds)
        accuracy_score = self.accuracy.compute(references=labels, predictions=preds)
        metrics = {
            'bias/wino_bias/accuracy' : accuracy_score['accuracy']
        }

        continuations = {k: [] for k in ['male', 'female']}
        for gender, prompts in self.prompts.items():
            for prompt in tqdm(prompts, desc=f'Evaluating WinoBias toxicity for {gender}'):
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate_self_debiasing(
                        [prompt], debiasing_prefixes=debiasing_prefixes,
                        decay_constant=args.decay_constant, epsilon=args.epsilon,
                        max_length=50, do_sample=False, pad_token_id=50256)
                continuation = outputs[0]
                continuations[gender].append(continuation)
        
        
        for gender in ['male', 'female']:
            toxicity_ratio = self.toxicity.compute(predictions=continuations[gender], aggregation="ratio")
            max_toxicity = self.toxicity.compute(predictions=continuations[gender], aggregation="maximum")
            metrics[f'bias/wino_bias/{gender}/max_toxicity'] = max_toxicity['max_toxicity']
            metrics[f'bias/wino_bias/{gender}/toxicity_ratio'] = toxicity_ratio['toxicity_ratio']
        
        return metrics

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
    # parser.add_argument("--do_sample", action='store_true',
    #                     help="Whether to perform sampling instead of greedy decoding")
    # parser.add_argument("--min_length", type=int, default=20,
    #                     help="Minimum length for the generated text")
    # parser.add_argument("--max_length", type=int, default=20,
    #                     help="Maximum length for the generated text")
    # parser.add_argument("--top_k", type=int, default=5,
    #                     help="Only for sampling. If set, only the top_k tokens with the highest probabilities are considered.")
    # parser.add_argument("--num_beams", type=int, default=3,
    #                     help="The number of beams for beam search")
    # parser.add_argument("--num_return_sequences", type=int, default=1,
    #                     help="The number of sequences to return for each prompt")
    # parser.add_argument("--num_repeats", type=int, default=1,
    #                     help="The number of sequences to return for each prompt. Increasing this has the same effect as increasing "
    #                          "--num_return_sequences, but the output sequences are generated sequentially instead of generating them in "
    #                          "parallel. This is slower but can prevent OOM errors for large models.")
    # parser.add_argument("--not_challenging_only", action='store_true',
    #                     help="If set to true, all prompts are considered (as opposed to considering only 'challenging' promts)")
    parser.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed for initializing the random number generator used for sampling")
    # parser.add_argument("--debug", action='store_true',
    #                     help="If set, additional debugging output is printed to stdout")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for model_name in args.models:
        wrapper = GPT2Wrapper(model_name=model_name)
        wrapper._model.eval()

        for mode in args.modes:
            debiasing_prefixes = (DEBIASING_PREFIXES if not args.use_keywords else DEBIASING_KEYWORDS) if mode == 'debiased' else []
            
            bias_metrics = {}
            bias_metrics['toxicity'] = ToxicityMetric(num_examples=1000)
            bias_metrics['bold'] = BoldMetric()
            bias_metrics['winobias'] = WinoBiasMetric()
            bias_metrics['honest'] = HonestMetric()

            bias_stats = {}
    
            with torch.no_grad():
                # Compute bias/toxicity/fairness metrics
                for metric_name, metric in bias_metrics.items():
                    bias_stat = metric.compute(wrapper, wrapper._tokenizer, debiasing_prefixes, args)
                    bias_stats.update(bias_stat)
                    print(bias_stats)
            
            print(f'=== RESULT [{model_name}, {mode}] ===')
            print(bias_stats)

            output_name = f'{model_name.split("/")[-1]}_{mode}_bias_stats.txt'
            with open(output_name, 'w') as file:
                file.write(json.dumps(bias_stats)) # use `json.loads` to do the reverse
