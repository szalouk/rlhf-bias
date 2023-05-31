from datasets import load_dataset
import evaluate
import random
from tqdm import tqdm

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
    
    def compute(self, ppo_trainer, tokenizer, max_length=50):
        model_continuations=[]

        # model = ppo_trainer.model
        device = ppo_trainer.accelerator.device

        for prompt in tqdm(self.toxic_prompts, desc='Evaluating Toxicity'):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = ppo_trainer.generate(inputs['input_ids'].squeeze(0), return_prompt=False, max_length=50, do_sample=False, pad_token_id=50256)
            all_outputs = ppo_trainer.accelerator.gather_for_metrics(outputs)
            print(f'all_outputs.shape = {all_outputs.shape}')
            # outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=50, do_sample=False, pad_token_id=50256)
            continuation = tokenizer.batch_decode(all_outputs)
            self.toxicity.add_batch(predictions=continuation)
            # model_continuations.extend(continuation)
            print(f'len(model_continuations) = {len(model_continuations)}')
            # model_continuations.append(continuation)
        
        toxicity_ratio = self.toxicity.compute(aggregation="ratio")
        max_toxicity = self.toxicity.compute(aggregation="maximum")
        # toxicity_ratio = self.toxicity.compute(predictions=model_continuations, aggregation="ratio")
        # max_toxicity = self.toxicity.compute(predictions=model_continuations, aggregation="maximum")

        metrics = {
            'bias/toxicity/max_toxicity' : max_toxicity['max_toxicity'],
            'bias/toxicity/toxicity_ratio' : toxicity_ratio['toxicity_ratio']
        }
        return metrics

