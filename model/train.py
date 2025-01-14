"""
Training script for the code generator model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
from pathlib import Path
from transformer import CodeGeneratorTransformer

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, 
                 optimizer, scheduler, device, save_dir):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (src, tgt) in enumerate(self.train_dataloader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt[:-1])
            loss = self.criterion(output.reshape(-1, output.size(-1)), 
                                tgt[1:].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 200 == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                print(f'Train Batch: {batch_idx:>5}, Loss: {loss.item():>10.4f}, '
                      f'Ms/Batch: {ms_per_batch:>10.4f}')

        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in self.val_dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                output = self.model(src, tgt[:-1])
                loss = self.criterion(output.reshape(-1, output.size(-1)), 
                                   tgt[1:].reshape(-1))
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def train(self, epochs, save_every=5):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 20)
            
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pt')
            
            if (epoch + 1) % save_every == 0:
                self.save_model(f'model_epoch_{epoch+1}.pt')
            
            self.scheduler.step()

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, self.save_dir / filename)

def main():
    # Hyperparameters
    vocab_size = 50000  # Adjust based on your tokenizer
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    epochs = 50
    learning_rate = 0.0001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = CodeGeneratorTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding index
    
    # TODO: Initialize your dataloaders
    train_dataloader = None  # Replace with your actual dataloader
    val_dataloader = None    # Replace with your actual dataloader
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir='checkpoints'
    )
    
    # Start training
    trainer.train(epochs=epochs)

if __name__ == '__main__':
    main()
