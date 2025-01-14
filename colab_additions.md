# Additional Colab Notebook Sections

## 1. Data Augmentation

```python
from utils.augmentation import CodeAugmenter

# Initialize augmenter
augmenter = CodeAugmenter(p=0.5)

class AugmentedCodeDataset(Dataset):
    def __init__(self, split='train', augment=True):
        self.dataset = load_dataset('codeparrot/codeparrot-clean', split=split)
        self.tokenizer = tokenizer
        self.max_length = 512
        self.augment = augment
        self.augmenter = CodeAugmenter() if augment else None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        code = item['code']
        
        # Apply augmentation in training
        if self.augment:
            code = self.augmenter.augment(code)
        
        # Tokenize
        inputs = self.tokenizer(
            item['prompt'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        outputs = self.tokenizer(
            code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': outputs['input_ids'].squeeze()
        }

# Create augmented datasets
train_dataset = AugmentedCodeDataset('train', augment=True)
val_dataset = AugmentedCodeDataset('validation', augment=False)
```

## 2. Performance Metrics

```python
from utils.metrics import CodeMetrics

# Initialize metrics
metrics = CodeMetrics()

# Start emissions tracking
emissions_tracker = metrics.track_carbon_emissions()

def evaluate_epoch(model, val_loader):
    """Evaluate model performance."""
    model.eval()
    all_metrics = {}
    
    # Get predictions
    predictions = []
    references = []
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        predictions.extend(preds)
        references.extend(refs)
    
    # Compute metrics
    all_metrics['bleu'] = metrics.compute_bleu(predictions, references)
    all_metrics.update(metrics.compute_rouge(predictions, references))
    
    # Syntax check
    syntax_correct = sum(metrics.compute_syntax_correctness(p) for p in predictions)
    all_metrics['syntax_accuracy'] = syntax_correct / len(predictions)
    
    # Code similarity
    similarities = [metrics.compute_code_similarity(p, r) 
                   for p, r in zip(predictions, references)]
    all_metrics['code_similarity'] = np.mean(similarities)
    
    # Resource usage
    all_metrics.update(metrics.track_resource_usage())
    
    return all_metrics

# Add to training loop
for epoch in range(num_epochs):
    # ... existing training code ...
    
    # Evaluate
    eval_metrics = evaluate_epoch(model, val_loader)
    
    # Log metrics
    wandb.log({
        'train_loss': total_loss/len(train_loader),
        'val_loss': val_loss,
        'bleu_score': eval_metrics['bleu'],
        'rouge_l': eval_metrics['rougeL'],
        'syntax_accuracy': eval_metrics['syntax_accuracy'],
        'code_similarity': eval_metrics['code_similarity'],
        'gpu_memory': eval_metrics.get('gpu_memory_percent', 0),
        'cpu_percent': eval_metrics['cpu_percent'],
        'epoch': epoch
    })

# Stop emissions tracking
total_emissions = emissions_tracker.stop()
print(f"Total CO2 emissions: {total_emissions:.2f}kg")
```

## 3. Extended Code Generation Examples

```python
from examples.code_examples import CODING_EXAMPLES, get_example_by_category

def test_model_generation():
    """Test model on various coding tasks."""
    model.eval()
    
    for category in CODING_EXAMPLES:
        print(f"\n=== Testing {category['category']} ===")
        
        for example in category['examples']:
            prompt = example['prompt']
            reference = example['reference']
            
            print(f"\nPrompt: {prompt}")
            print("\nGenerated Code:")
            generated = generate_code(prompt)
            print(generated)
            
            print("\nMetrics:")
            similarity = metrics.compute_code_similarity(generated, reference)
            syntax_ok = metrics.compute_syntax_correctness(generated)
            
            print(f"Code Similarity: {similarity:.2f}")
            print(f"Syntax Correct: {'✓' if syntax_ok else '✗'}")
            print("-" * 50)

# Test specific categories
test_cases = {
    "Basic Functions": [
        "Write a function to calculate the fibonacci sequence",
        "Create a function to find prime numbers up to n",
        "Implement a function to reverse a linked list"
    ],
    "Data Structures": [
        "Implement a min heap class",
        "Create a trie data structure for string storage",
        "Implement a LRU cache"
    ],
    "Algorithms": [
        "Write a function to find the longest increasing subsequence",
        "Implement Dijkstra's shortest path algorithm",
        "Create a function for matrix chain multiplication"
    ]
}

def test_specific_cases():
    """Test model on specific challenging cases."""
    model.eval()
    
    for category, prompts in test_cases.items():
        print(f"\n=== Testing {category} ===")
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            # Generate with different parameters
            print("\n1. Basic Generation:")
            print(generate_code(prompt))
            
            print("\n2. With Temperature 0.7:")
            print(generate_code(prompt, temperature=0.7))
            
            print("\n3. With Beam Search (num_beams=5):")
            print(generate_code(prompt, num_beams=5))
            
            print("-" * 50)

# Run tests
test_model_generation()
test_specific_cases()
```

## 4. Training Progress Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_progress(metrics_history):
    """Plot training metrics."""
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # BLEU score
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['bleu_score'])
    plt.title('BLEU Score')
    
    # Syntax accuracy
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['syntax_accuracy'])
    plt.title('Syntax Accuracy')
    
    # Resource usage
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['gpu_memory'], label='GPU Memory')
    plt.plot(metrics_history['cpu_percent'], label='CPU Usage')
    plt.title('Resource Usage')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot progress
plot_training_progress(metrics_history)
```

Add these sections to your Colab notebook for:
1. Advanced data augmentation with 6 different techniques
2. Comprehensive performance metrics including BLEU, ROUGE, syntax accuracy
3. Extended code generation examples with different parameters
4. Visual training progress monitoring
5. Carbon emissions tracking

The augmentation techniques include:
- Variable renaming
- Binary operand swapping
- Comment addition
- Case style changes
- Type hint insertion
- Function reordering

The metrics now track:
- Code similarity
- Syntax correctness
- BLEU and ROUGE scores
- Resource usage (CPU, GPU, Memory)
- Carbon emissions

Would you like me to:
1. Add more augmentation techniques?
2. Include additional test cases?
3. Add more visualization options?
