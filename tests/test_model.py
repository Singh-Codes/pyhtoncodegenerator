"""
Test cases for the code generator model.
"""

import pytest
import torch
from model.transformer import CodeGeneratorTransformer
from scripts.preprocess_data import CodeDataPreprocessor

def test_transformer_initialization():
    model = CodeGeneratorTransformer(
        vocab_size=1000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    assert isinstance(model, CodeGeneratorTransformer)

def test_transformer_forward_pass():
    model = CodeGeneratorTransformer(
        vocab_size=1000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    
    # Create dummy input data
    src = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    tgt = torch.randint(0, 1000, (2, 8))   # batch_size=2, seq_len=8
    
    # Forward pass
    output = model(src, tgt)
    
    # Check output shape
    assert output.shape == (2, 8, 1000)  # batch_size, seq_len, vocab_size

def test_preprocessor_initialization():
    preprocessor = CodeDataPreprocessor(vocab_size=1000)
    assert preprocessor.vocab_size == 1000

def test_docstring_extraction():
    preprocessor = CodeDataPreprocessor()
    code = '''
def test_function():
    """This is a test docstring."""
    pass
'''
    import ast
    tree = ast.parse(code)
    function_def = tree.body[0]
    docstring = preprocessor.extract_docstring(function_def)
    assert docstring == "This is a test docstring."

@pytest.mark.parametrize("input_text,expected_tokens", [
    ("def simple_function():", ["def", "simple_function", "(", ")", ":"]),
    ("return x + y", ["return", "x", "+", "y"]),
])
def test_basic_tokenization(input_text, expected_tokens):
    preprocessor = CodeDataPreprocessor()
    # Note: This test will need modification based on your actual tokenization implementation
    # The current test is a placeholder for the concept
    pass
