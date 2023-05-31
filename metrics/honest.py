from datasets import load_dataset
import evaluate
import random
from tqdm import tqdm

class HonestMetric:
    def __init__(self, num_examples=50):
        honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
        
        self.prompts = {}
        self.prompts['queer'] = random.sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('queer')], num_examples)
        self.prompts['nonqueer'] = random.sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('nonqueer')], num_examples)

        self.generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,  # Ignored.
            "top_p": 1.0,  # Ignored.
            "do_sample": True,  # If False, does greedy sampling.
            "eos_token_id": 100_000,
        }
        self.honest = evaluate.load('honest', 'en')
    
    def compute(self, ppo_trainer, tokenizer, num_generations=20):
        continuations = {k: [] for k in ['queer', 'nonqueer']}
        # model = ppo_trainer.model
        # device = ppo_trainer.accelerator.device

        self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        self.generation_kwargs["num_return_sequences"] = num_generations

        for gender, prompts in self.prompts.items():
            for prompt in tqdm(prompts, desc=f'Evaluating honest for {gender}'):
                inputs = tokenizer(prompt, return_tensors="pt")
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                max_len = inputs["input_ids"].shape[-1] + 10
                self.generation_kwargs["max_length"] = max_len
                outputs = ppo_trainer.generate(inputs['input_ids'], return_prompt=False, generation_kwargs=self.generation_kwargs)
                # outputs = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, max_length=max_len,
                #     num_return_sequences=num_generations, **self.generation_kwargs)
                continuation = tokenizer.decode(outputs[0])
                continuations[gender].append(continuation)
        
        groups = ['queer'] * 50 + ['nonqueer'] * 50
        continuations = [c.split() for c in continuations['queer']] + [q.split() for q in continuations['nonqueer']]

        honest_score = self.honest.compute(predictions=continuations, groups=groups)
        metrics = {
            'bias/honest/queer' : honest_score['honest_score_per_group']['queer'],
            'bias/honest/nonqueer' : honest_score['honest_score_per_group']['nonqueer']
        }
        return metrics

