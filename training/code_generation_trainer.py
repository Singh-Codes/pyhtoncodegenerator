"""
Training module for the code generator AI to handle various situations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

from model.transformer import TransformerModel
from error_resolution import ErrorResolver
from dependency_manager import DependencyManager

class CodeGenerationDataset(Dataset):
    def __init__(self, scenarios: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.scenarios = scenarios
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scenario = self.scenarios[idx]
        
        # Tokenize input prompt
        prompt_encoding = self.tokenizer(
            scenario['prompt'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target code
        target_encoding = self.tokenizer(
            scenario['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': prompt_encoding['input_ids'].squeeze(),
            'attention_mask': prompt_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

class CodeGenerationTrainer:
    def __init__(self, model: TransformerModel, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Initialize components
        self.error_resolver = ErrorResolver()
        self.dependency_manager = DependencyManager(".")
        
        # Configure logging
        logging.basicConfig(
            filename='logs/trainer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def generate_training_scenarios(self) -> List[Dict[str, str]]:
        """Generate diverse training scenarios."""
        scenarios = []
        
        # Basic function scenarios
        scenarios.extend([
            {
                'prompt': 'Write a function to calculate the factorial of a number',
                'code': '''def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)'''
            },
            {
                'prompt': 'Create a function to check if a string is a palindrome',
                'code': '''def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]'''
            }
        ])
        
        # Error handling scenarios
        scenarios.extend([
            {
                'prompt': 'Write a function that safely divides two numbers and handles division by zero',
                'code': '''def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"
    except TypeError:
        return "Invalid input types"'''
            }
        ])
        
        # API integration scenarios
        scenarios.extend([
            {
                'prompt': 'Create a function to fetch data from a REST API',
                'code': '''import requests

def fetch_api_data(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return f"Error fetching data: {str(e)}"'''
            }
        ])
        
        # Database scenarios
        scenarios.extend([
            {
                'prompt': 'Write a function to safely connect to a SQLite database',
                'code': '''import sqlite3
from contextlib import contextmanager

@contextmanager
def database_connection(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        yield conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()'''
            }
        ])
        
        # File handling scenarios
        scenarios.extend([
            {
                'prompt': 'Create a function to safely read and write JSON files',
                'code': '''import json
from pathlib import Path

def handle_json_file(file_path, data=None):
    try:
        path = Path(file_path)
        if data:  # Write mode
            with path.open('w') as f:
                json.dump(data, f, indent=4)
            return True
        else:  # Read mode
            with path.open('r') as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error handling JSON file: {e}")
        return None'''
            }
        ])
        
        # Dependency management scenarios
        scenarios.extend([
            {
                'prompt': 'Write a function to check and install required Python packages',
                'code': '''import subprocess
import sys

def ensure_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ])'''
            }
        ])
        
        # Web development scenarios
        scenarios.extend([
            {
                'prompt': 'Create a basic Flask route with error handling',
                'code': '''from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/<int:id>', methods=['GET'])
def get_item(id):
    try:
        # Simulate database query
        if id <= 0:
            raise ValueError("Invalid ID")
        return jsonify({"id": id, "status": "success"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500'''
            }
        ])
        
        # Data processing scenarios
        scenarios.extend([
            {
                'prompt': 'Write a function to process CSV data using pandas',
                'code': '''import pandas as pd
from typing import Union, List

def process_csv(
    file_path: str,
    columns: List[str] = None,
    filters: dict = None
) -> Union[pd.DataFrame, None]:
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Apply column selection
        if columns:
            df = df[columns]
            
        # Apply filters
        if filters:
            for column, value in filters.items():
                df = df[df[column] == value]
                
        return df
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None'''
            }
        ])
        
        return scenarios

    def train(self, 
              train_scenarios: List[Dict[str, str]], 
              batch_size: int = 8,
              num_epochs: int = 10,
              learning_rate: float = 1e-4):
        """Train the model on various scenarios."""
        try:
            # Create dataset and dataloader
            dataset = CodeGenerationDataset(train_scenarios, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize optimizer and loss function
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                
                for batch in dataloader:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                avg_loss = total_loss / len(dataloader)
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def evaluate_scenario(self, prompt: str) -> Dict[str, Any]:
        """Evaluate the model on a specific scenario."""
        try:
            self.model.eval()
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate code
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=512,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2
                )
            
            # Decode generated code
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Test the generated code
            try:
                exec(generated_code)
                status = "success"
                error = None
            except Exception as e:
                status = "error"
                error = str(e)
                
                # Try to resolve error
                if status == "error":
                    solution = self.error_resolver.handle_error(e, generated_code)
                    if solution:
                        status = "resolved"
                        error = None
                        generated_code = solution['solution']
            
            return {
                'status': status,
                'code': generated_code,
                'error': error
            }
            
        except Exception as e:
            logging.error(f"Error evaluating scenario: {str(e)}")
            return {
                'status': 'error',
                'code': None,
                'error': str(e)
            }

    def fine_tune_on_errors(self, error_cases: List[Dict[str, str]]):
        """Fine-tune the model on error cases to improve error handling."""
        try:
            # Prepare error cases for training
            error_scenarios = []
            for case in error_cases:
                error_scenarios.append({
                    'prompt': f"Fix code with error: {case['error']}\nCode:\n{case['code']}",
                    'code': case['solution']
                })
            
            # Fine-tune on error cases
            self.train(error_scenarios, batch_size=4, num_epochs=5)
            
        except Exception as e:
            logging.error(f"Error during fine-tuning: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save the trained model and tokenizer."""
        try:
            # Create directory if it doesn't exist
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(save_path / "model")
            self.tokenizer.save_pretrained(save_path / "tokenizer")
            
            logging.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
