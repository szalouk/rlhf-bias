from datasets import load_dataset
import evaluate
import random
from tqdm import tqdm

class BoldMetric:
    def __init__(self, num_examples=50):
        bold = load_dataset("AlexaAI/bold", split="train")
        female_bold = (random.sample([p for p in bold if p['category'] == 'American_actresses'], num_examples))
        male_bold = (random.sample([p for p in bold if p['category'] == 'American_actors'], num_examples))

        self.prompts = {}
        self.prompts['male'] = [p['prompts'][0] for p in male_bold]
        self.prompts['female'] = [p['prompts'][0] for p in female_bold]
        self.regard = evaluate.load('regard', 'compare')
    
    def compute(self, ppo_trainer, tokenizer, max_length=50):
        continuations = {k: [] for k in ['male', 'female']}

        # model = ppo_trainer.model
        # device = ppo_trainer.accelerator.device

        generation_kwargs = {
            "max_length": 50,
            "do_sample": False,  # If False, does greedy sampling.
            "pad_token_id": 50256
        }

        for gender, prompts in self.prompts.items():
            for prompt in tqdm(prompts, desc=f'Evaluating Bold for {gender}'):
                inputs = tokenizer(prompt, return_tensors="pt")
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = ppo_trainer.generate(inputs['input_ids'], return_prompt=False, generation_kwargs=generation_kwargs)
                # outputs = model.generate(**inputs, max_length=50, do_sample=False, pad_token_id=50256)
                continuation = tokenizer.decode(outputs[0])
                continuations[gender].append(continuation)
        
        regard = self.regard.compute(data=continuations['male'], references=continuations['female'])

        metrics = {
            'bias/bold/regard_difference/positive' : regard['regard_difference']['positive'],
            'bias/bold/regard_difference/neutral' : regard['regard_difference']['neutral'],
            'bias/bold/regard_difference/other' : regard['regard_difference']['other'],
            'bias/bold/regard_difference/negative' : regard['regard_difference']['negative']
        }
        return metrics

