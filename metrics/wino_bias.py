from datasets import load_dataset
import evaluate
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

class WinoBiasMetric:
    def __init__(self):
        data = load_dataset("mathemakitten/winobias_antistereotype_test", split="test")
        
        self.prompts1 = [data[i]['text'] + data[i]["classes"][0] for i in range(len(data))]
        self.prompts2 = [data[i]['text'] + data[i]["classes"][1] for i in range(len(data))]

        self.accuracy = evaluate.load("accuracy")
    
    def eval_loss(self, prompt, model, tokenizer, device):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs[1].detach().cpu().item()
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

    def compute(self, ppo_trainer, tokenizer):
        model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
        device = ppo_trainer.accelerator.device

        preds = []
        for prompt1, prompt2 in tqdm(zip(self.prompts1, self.prompts2), desc='Evaluating WinoBias'):
            loss1 = self.eval_loss(prompt1, model, tokenizer, device)
            loss2 = self.eval_loss(prompt2, model, tokenizer, device)
            pred = 0 if loss1 < loss2 else 1
            preds.append(pred)

        labels = [0] * len(preds)
        accuracy_score = self.accuracy.compute(references=labels, predictions=preds)

        metrics = {
            'bias/wino_bias/accuracy' : accuracy_score['accuracy']
        }
        return metrics

