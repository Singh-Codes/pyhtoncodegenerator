"""
Data preprocessing script for code generation model.
"""

import ast
import tokenize
from pathlib import Path
import json
from typing import List, Tuple, Dict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class CodeDataPreprocessor:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def extract_docstring(self, node: ast.AST) -> str:
        """Extract docstring from an AST node."""
        return ast.get_docstring(node) or ""

    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single Python file and extract code-docstring pairs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            pairs = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = self.extract_docstring(node)
                    if docstring:
                        code = ast.unparse(node)
                        pairs.append({
                            'description': docstring,
                            'code': code
                        })
            
            return pairs
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def train_tokenizer(self, code_samples: List[str]):
        """Train a BPE tokenizer on the code samples."""
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        )
        
        # Pre-tokenize using whitespace
        tokenizer.pre_tokenizer = Whitespace()
        
        # Train the tokenizer
        tokenizer.train_from_iterator(code_samples, trainer=trainer)
        
        self.tokenizer = tokenizer

    def save_tokenizer(self, path: str):
        """Save the trained tokenizer."""
        if self.tokenizer:
            self.tokenizer.save(path)

    def load_tokenizer(self, path: str):
        """Load a trained tokenizer."""
        self.tokenizer = Tokenizer.from_file(path)

    def tokenize_data(self, text: str) -> List[int]:
        """Tokenize text using the trained tokenizer."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load a tokenizer first.")
        
        return self.tokenizer.encode(text).ids

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Process all Python files in a directory and its subdirectories."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_pairs = []
        code_samples = []
        
        # Process all Python files
        for py_file in input_dir.rglob('*.py'):
            pairs = self.process_file(py_file)
            all_pairs.extend(pairs)
            code_samples.extend(pair['code'] for pair in pairs)
        
        # Train tokenizer if not already trained
        if not self.tokenizer:
            print("Training tokenizer...")
            self.train_tokenizer(code_samples)
            self.save_tokenizer(str(output_dir / 'code_tokenizer.json'))
        
        # Tokenize and save the processed data
        processed_data = []
        for pair in all_pairs:
            try:
                description_tokens = self.tokenize_data(pair['description'])
                code_tokens = self.tokenize_data(pair['code'])
                
                processed_data.append({
                    'description_tokens': description_tokens,
                    'code_tokens': code_tokens,
                    'original_description': pair['description'],
                    'original_code': pair['code']
                })
            except Exception as e:
                print(f"Error tokenizing pair: {str(e)}")
        
        # Save processed data
        with open(output_dir / 'processed_data.json', 'w') as f:
            json.dump(processed_data, f)

def main():
    # Initialize preprocessor
    preprocessor = CodeDataPreprocessor()
    
    # Set up directories
    input_dir = Path('data/raw_data')
    output_dir = Path('data/processed_data')
    
    # Process the data
    preprocessor.process_directory(input_dir, output_dir)

if __name__ == '__main__':
    main()
