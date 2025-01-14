"""
Local testing script for the transformer model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.transformer import CodeGeneratorTransformer
import json
import os
from pathlib import Path

class CodeDataset(Dataset):
    """Simple dataset for testing the model locally."""
    def __init__(self, data_dir: str = "data/test"):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        """Load test data. If no data exists, create simple examples."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
            # Create simple test data
            test_data = [
                {
                    "input": "Create a function to add two numbers",
                    "output": "def add_numbers(a, b):\n    return a + b"
                },
                {
                    "input": "Write a function to check if string is palindrome",
                    "output": "def is_palindrome(s):\n    return s == s[::-1]"
                }
            ]
            with open(self.data_dir / "test_data.json", "w") as f:
                json.dump(test_data, f, indent=2)
            
            self.samples = test_data
        else:
            with open(self.data_dir / "test_data.json", "r") as f:
                self.samples = json.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input": sample["input"],
            "output": sample["output"]
        }

def test_model():
    """Test the transformer model locally."""
    # Initialize model with vocab size
    model = CodeGeneratorTransformer(vocab_size=10000)
    
    # Create test dataset
    dataset = CodeDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Test forward pass
    for batch in dataloader:
        inputs = batch["input"]
        outputs = batch["output"]
        
        # Convert to tensors (simplified for testing)
        input_tokens = torch.randint(0, 1000, (2, 20))  # Match sequence length
        output_tokens = torch.randint(0, 1000, (2, 20))  # Match sequence length
        
        # Forward pass
        try:
            pred = model(input_tokens, output_tokens)
            print("Forward pass successful!")
            print(f"Input shape: {input_tokens.shape}")
            print(f"Output shape: {pred.shape}")
            break
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            raise e

def train_locally(num_epochs=2):
    """Train the model locally for a few epochs."""
    # Initialize model and optimizer
    model = CodeGeneratorTransformer(vocab_size=10000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset
    dataset = CodeDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Simulate tokenized data
            input_tokens = torch.randint(0, 1000, (2, 10))
            output_tokens = torch.randint(0, 1000, (2, 20))
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_tokens, output_tokens[:, :-1])
            
            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), 
                output_tokens[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model_local.pt")
    print("Model saved to checkpoints/model_local.pt")

if __name__ == "__main__":
    print("Testing model...")
    test_model()
    
    print("\nTraining model locally...")
    train_locally()
