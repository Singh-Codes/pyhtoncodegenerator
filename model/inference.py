"""
Inference script for code generation.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional
from tokenizers import Tokenizer
from transformer import CodeGeneratorTransformer

class CodeGenerator:
    def __init__(self, model_path: str, tokenizer_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = CodeGeneratorTransformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_code(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate code based on the input prompt.
        
        Args:
            prompt (str): Natural language description of the code to generate
            max_length (int): Maximum length of generated code
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            str: Generated code
        """
        # Encode the prompt
        encoded = self.tokenizer.encode(prompt)
        src_tokens = torch.tensor([encoded.ids]).to(self.device)
        
        # Initialize target with BOS token
        tgt_tokens = torch.tensor([[self.tokenizer.token_to_id("[BOS]")]]).to(self.device)
        
        # Generate tokens one by one
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(src_tokens)
            
            for _ in range(max_length):
                # Get model predictions
                output = self.model.decode(tgt_tokens, memory)
                next_token_logits = output[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the new token
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
                
                # Check if we've generated an EOS token
                if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                    break
        
        # Decode the generated tokens
        generated_ids = tgt_tokens[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        # Clean up the generated text
        generated_text = self._clean_generated_code(generated_text)
        
        return generated_text

    def _clean_generated_code(self, code: str) -> str:
        """Clean up the generated code by removing special tokens and fixing indentation."""
        # Remove special tokens
        code = code.replace("[BOS]", "").replace("[EOS]", "").strip()
        
        # Fix indentation
        lines = code.split('\n')
        cleaned_lines = []
        current_indent = 0
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            # Adjust indentation based on code structure
            if stripped_line.startswith(('def ', 'class ', 'if ', 'for ', 'while ')):
                cleaned_lines.append('    ' * current_indent + stripped_line)
                current_indent += 1
            elif stripped_line.startswith(('return', 'break', 'continue')):
                current_indent = max(0, current_indent - 1)
                cleaned_lines.append('    ' * current_indent + stripped_line)
            else:
                cleaned_lines.append('    ' * current_indent + stripped_line)
        
        return '\n'.join(cleaned_lines)

def main():
    # Initialize generator
    generator = CodeGenerator(
        model_path='checkpoints/best_model.pt',
        tokenizer_path='data/processed_data/code_tokenizer.json'
    )
    
    # Example prompts
    prompts = [
        "Write a function to calculate the factorial of a number",
        "Create a class for a binary search tree with insert and search methods",
        "Write a function to check if a string is a palindrome"
    ]
    
    # Generate code for each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("\nGenerated Code:")
        generated_code = generator.generate_code(prompt)
        print(generated_code)
        print("-" * 80)

if __name__ == '__main__':
    main()
