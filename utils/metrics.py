"""
Performance metrics for code generation.
"""

import torch
import numpy as np
from typing import List, Dict
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from codecarbon import EmissionsTracker
import psutil
import GPUtil

class CodeMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        self.emissions_tracker = EmissionsTracker()

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        for pred, ref in zip(predictions, references):
            result = self.rouge_scorer.score(pred, ref)
            scores['rouge1'] += result['rouge1'].fmeasure
            scores['rouge2'] += result['rouge2'].fmeasure
            scores['rougeL'] += result['rougeL'].fmeasure
        
        n = len(predictions)
        return {k: v/n for k, v in scores.items()}

    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score."""
        return self.bleu.corpus_score(predictions, [references]).score

    def compute_code_similarity(self, generated: str, reference: str) -> float:
        """Compute code similarity using token-based comparison."""
        def tokenize_code(code: str) -> List[str]:
            import tokenize
            from io import StringIO
            tokens = []
            try:
                for tok in tokenize.generate_tokens(StringIO(code).readline):
                    if tok.type not in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE]:
                        tokens.append(tok.string)
            except:
                tokens = code.split()
            return tokens
        
        gen_tokens = set(tokenize_code(generated))
        ref_tokens = set(tokenize_code(reference))
        
        if not ref_tokens:
            return 0.0
        
        intersection = gen_tokens.intersection(ref_tokens)
        return len(intersection) / len(ref_tokens)

    def compute_syntax_correctness(self, code: str) -> bool:
        """Check if generated code is syntactically correct."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except:
            return False

    def track_resource_usage(self) -> Dict[str, float]:
        """Track computational resources."""
        metrics = {}
        
        # CPU usage
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['memory_percent'] = psutil.virtual_memory().percent
        
        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_memory_percent'] = gpus[0].memoryUtil * 100
                metrics['gpu_load_percent'] = gpus[0].load * 100
        except:
            pass
        
        return metrics

    def track_carbon_emissions(self):
        """Track carbon emissions during training."""
        self.emissions_tracker.start()
        return self.emissions_tracker

    def stop_tracking_emissions(self) -> float:
        """Stop tracking emissions and return total."""
        return self.emissions_tracker.stop()

    def compute_perplexity(self, model, dataloader, device) -> float:
        """Compute model perplexity."""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=0  # Assuming 0 is pad token
                )
                
                total_loss += loss.item() * labels[:, 1:].ne(0).sum().item()
                total_tokens += labels[:, 1:].ne(0).sum().item()
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    def evaluate_model(self, 
                      model, 
                      test_dataloader, 
                      tokenizer, 
                      device,
                      num_examples=100) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        model.eval()
        metrics = {}
        predictions = []
        references = []
        
        # Generate predictions
        for i, batch in enumerate(test_dataloader):
            if i >= num_examples:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode predictions and references
            pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(pred)
            references.extend(ref)
        
        # Compute metrics
        metrics['bleu'] = self.compute_bleu(predictions, references)
        rouge_scores = self.compute_rouge(predictions, references)
        metrics.update(rouge_scores)
        
        # Syntax correctness
        correct_syntax = sum(self.compute_syntax_correctness(p) for p in predictions)
        metrics['syntax_accuracy'] = correct_syntax / len(predictions)
        
        # Code similarity
        similarities = [self.compute_code_similarity(p, r) 
                       for p, r in zip(predictions, references)]
        metrics['code_similarity'] = np.mean(similarities)
        
        # Perplexity
        metrics['perplexity'] = self.compute_perplexity(model, test_dataloader, device)
        
        return metrics
