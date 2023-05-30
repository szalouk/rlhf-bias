from datasets import load_dataset
import evaluate
import random

class BoldMetric:
    def __init__(self, num_examples=50):
        bold = load_dataset("AlexaAI/bold", split="train")
        female_bold = (random.sample([p for p in bold if p['category'] == 'American_actresses'], num_examples))
        male_bold = (random.sample([p for p in bold if p['category'] == 'American_actors'], num_examples))

        self.prompts = {}
        self.prompts['male'] = [p['prompts'][0] for p in male_bold]
        self.prompts['female'] = [p['prompts'][0] for p in female_bold]
        self.regard = evaluate.load('regard', 'compare')
    
    def compute(self, model, tokenizer, max_length=50):
        continuations = {k: [] for k in ['male', 'female']}

        for gender, prompts in self.prompts.items():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=50, do_sample=False, pad_token_id=50256)
                continuation = tokenizer.decode(outputs[0]).replace(prompt,'')
                continuations[gender].append(continuation)
        
        regard = self.regard.compute(data=continuations['male'], references=continuations['female'])

        return {'regard': regard}

