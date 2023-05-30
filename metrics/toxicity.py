from datasets import load_dataset
import evaluate
import random

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
    
    def compute(self, model, tokenizer, max_length=50):
        model_continuations=[]
        for prompt in self.toxic_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=50, do_sample=False, pad_token_id=50256)
            continuation = tokenizer.decode(outputs[0]).replace(prompt,'')
            model_continuations.append(continuation)
        
        toxicity_ratio = self.toxicity.compute(predictions=model_continuations, aggregation="ratio")
        max_toxicity = self.toxicity.compute(predictions=model_continuations, aggregation="maximum")

        metrics = {
            'bias/toxicity/max_toxicity' : max_toxicity['max_toxicity'],
            'bias/toxicity/toxicity_ratio' : toxicity_ratio['toxicity_ratio']
        }
        return metrics

