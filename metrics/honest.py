from datasets import load_dataset
import evaluate
import random

class HonestMetric:
    def __init__(self, num_examples=50):
        honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
        
        self.prompts = {}
        self.prompts['queer'] = random.sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('queer')], num_examples)
        self.prompts['nonqueer'] = random.sample([h['template_masked'].replace(' [M].','') for h in honest_dataset if h['category'].startswith('nonqueer')], num_examples)

        self.honest = evaluate.load('honest', 'en')
    
    def compute(self, model, tokenizer, generation_kwargs, num_generations=20):
        continuations = {k: [] for k in ['queer', 'nonqueer']}

        for gender, prompts in self.prompts.items():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                max_len = len(inputs['input_ids']) + 10
                outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=max_len,
                    num_return_sequences=num_generations, **generation_kwargs)
                continuation = tokenizer.decode(outputs[0]).replace(prompt,'')
                continuations[gender].append(continuation)
        
        groups = ['queer'] * 50 + ['nonqueer'] * 50
        continuations = [c.split() for c in continuations['queer']] + [q.split() for q in continuations['nonqueer']]

        honest_score = self.honest.compute(predictions=continuations, groups=groups)
        metrics = {
            'bias/honest/queer' : honest_score['honest_score_per_group']['queer'],
            'bias/honest/nonqueer' : honest_score['honest_score_per_group']['nonqueer']
        }
        return metrics

