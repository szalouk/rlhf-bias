from datasets import load_dataset
import evaluate
import random

class WinoBiasMetric:
    def __init__(self):
        wino_bias = load_dataset("mathemakitten/winobias_antistereotype_test", split="test")
        
        self.prompts1 = [data[i]['text'] + data[i]["classes"][0] for i in range(len(data))]
        self.prompts2 = [data[i]['text'] + data[i]["classes"][1] for i in range(len(data))]

        self.accuracy = evaluate.load("accuracy")
    
    def eval_loss(prompt, model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model(**input, labels=input["input_ids"])
        loss = outputs[1].detach().item()
        return loss

    def compute(model, tokenizer):
        continuations = {k: [] for k in ['queer', 'nonqueer']}

        preds = []
        for prompt1, prompt2 in zip(self.prompts1, self.prompts2):
            loss1 = self.eval_loss(prompt1, model, tokenizer)
            loss2 = self.eval_loss(prompt2, model, tokenizer)
            pred = 0 if loss1 < loss2 else 1
            preds.append(pred)

        labels = [0] * len(preds)
        accuracy_score = self.accuracy.compute(references=labels, predictions=preds)

        return {'wino_bias_acc': accuracy_score}

