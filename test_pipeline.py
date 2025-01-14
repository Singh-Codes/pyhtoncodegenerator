"""Test the entire training pipeline locally before Colab."""
import torch
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from utils.augmentation import CodeAugmenter
from utils.metrics import CodeMetrics
from examples.code_examples import get_example_by_category
from model.transformer import CodeGeneratorTransformer

def test_data_pipeline():
    """Test data loading and augmentation."""
    print("\n=== Testing Data Pipeline ===")
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    augmenter = CodeAugmenter()
    
    # Test code augmentation
    test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    augmented = augmenter.augment(test_code)
    print("Original code:")
    print(test_code)
    print("\nAugmented code:")
    print(augmented)
    
    # Test tokenization
    tokens = tokenizer(test_code, return_tensors='pt')
    print("\nTokenization successful:", all(len(v.shape) == 2 for v in tokens.values()))

def test_model_pipeline():
    """Test model forward pass."""
    print("\n=== Testing Model Pipeline ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = CodeGeneratorTransformer(
        vocab_size=50265,  # RoBERTa tokenizer vocab size
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    dummy_input = torch.randint(0, 50265, (batch_size, seq_length)).to(device)
    dummy_target = torch.randint(0, 50265, (batch_size, seq_length)).to(device)
    
    try:
        output = model(dummy_input, dummy_target)
        print("Model forward pass successful")
        print("Output shape:", output.shape)
    except Exception as e:
        print("Model forward pass failed:", str(e))

def test_metrics():
    """Test metrics computation."""
    print("\n=== Testing Metrics ===")
    metrics = CodeMetrics()
    
    # Test code
    generated = "def add(a, b):\n    return a + b"
    reference = "def add(x, y):\n    return x + y"
    
    # Test individual metrics
    print("Testing BLEU score...")
    bleu = metrics.compute_bleu([generated], [reference])
    print(f"BLEU score: {bleu}")
    
    print("\nTesting ROUGE scores...")
    rouge_scores = metrics.compute_rouge([generated], [reference])
    print("ROUGE scores:", rouge_scores)
    
    print("\nTesting syntax check...")
    syntax_ok = metrics.compute_syntax_correctness(generated)
    print(f"Syntax correct: {syntax_ok}")
    
    print("\nTesting code similarity...")
    similarity = metrics.compute_code_similarity(generated, reference)
    print(f"Code similarity: {similarity}")
    
    print("\nTesting resource tracking...")
    resources = metrics.track_resource_usage()
    print("Resource usage:", resources)

def main():
    """Run all tests."""
    try:
        test_data_pipeline()
        test_model_pipeline()
        test_metrics()
        print("\n✅ All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")

if __name__ == "__main__":
    main()
