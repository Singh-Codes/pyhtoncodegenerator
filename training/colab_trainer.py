"""
Google Colab training module for efficient resource utilization.
"""

import os
import sys
import json
import time
import psutil
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from google.colab import drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

@dataclass
class ResourceMetrics:
    """Tracks resource usage during training."""
    ram_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    gpu_memory: Optional[float]
    training_time: float
    batch_size: int
    learning_rate: float
    performance_metrics: Dict[str, float]

class ColabResourceManager:
    """Manages Colab resource allocation and monitoring."""
    
    def __init__(self):
        self.metrics_history: List[ResourceMetrics] = []
        self.drive = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for resource monitoring."""
        logging.basicConfig(
            filename='logs/colab_training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def mount_drive(self):
        """Mount Google Drive for checkpoint storage."""
        try:
            drive.mount('/content/drive')
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()
            self.drive = GoogleDrive(gauth)
            logging.info("Successfully mounted Google Drive")
            return True
        except Exception as e:
            logging.error(f"Error mounting Google Drive: {str(e)}")
            return False

    def check_gpu_availability(self) -> bool:
        """Check if GPU is available in Colab."""
        try:
            return torch.cuda.is_available()
        except:
            return False

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        metrics = {
            'ram_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'gpu_usage': None,
            'gpu_memory': None
        }
        
        if self.check_gpu_availability():
            try:
                gpu = torch.cuda.get_device_properties(0)
                metrics['gpu_memory'] = torch.cuda.memory_allocated(0) / gpu.total_memory * 100
                metrics['gpu_usage'] = torch.cuda.utilization()
            except:
                pass
                
        return metrics

    def optimize_batch_size(self, initial_batch_size: int) -> int:
        """Dynamically adjust batch size based on available resources."""
        metrics = self.get_resource_usage()
        
        if metrics['ram_percent'] > 90 or (metrics['gpu_memory'] and metrics['gpu_memory'] > 90):
            return max(1, initial_batch_size // 2)
        elif metrics['ram_percent'] < 50 and (not metrics['gpu_memory'] or metrics['gpu_memory'] < 50):
            return initial_batch_size * 2
            
        return initial_batch_size

class ColabTrainer:
    """Manages model training on Google Colab."""
    
    def __init__(self, model: nn.Module, resource_manager: ColabResourceManager):
        self.model = model
        self.resource_manager = resource_manager
        self.checkpoint_dir = Path('/content/drive/MyDrive/model_checkpoints')
        self.metrics_history = []
        
    def prepare_training_environment(self):
        """Set up the training environment."""
        # Mount Google Drive
        if not self.resource_manager.mount_drive():
            raise RuntimeError("Failed to mount Google Drive")
            
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = 'cuda' if self.resource_manager.check_gpu_availability() else 'cpu'
        self.model.to(self.device)

    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float):
        """Save model checkpoint to Google Drive."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            # Save metrics
            metrics_path = self.checkpoint_dir / f'metrics_epoch_{epoch}.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history[-1].__dict__, f)
                
            logging.info(f"Saved checkpoint for epoch {epoch}")
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer) -> bool:
        """Load model checkpoint from Google Drive."""
        try:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            if not checkpoint_path.exists():
                return False
                
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load metrics
            metrics_path = self.checkpoint_dir / f'metrics_epoch_{epoch}.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics_dict = json.load(f)
                    self.metrics_history.append(ResourceMetrics(**metrics_dict))
                    
            logging.info(f"Loaded checkpoint from epoch {epoch}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return False

    def train(self, 
              train_loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              criterion: nn.Module,
              num_epochs: int,
              initial_batch_size: int = 32,
              checkpoint_frequency: int = 1):
        """Train the model with resource optimization."""
        try:
            self.prepare_training_environment()
            
            start_epoch = 0
            # Try to load latest checkpoint
            for epoch in range(num_epochs - 1, -1, -1):
                if self.load_checkpoint(epoch, optimizer):
                    start_epoch = epoch + 1
                    break
            
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                self.model.train()
                total_loss = 0
                
                # Optimize batch size
                current_batch_size = self.resource_manager.optimize_batch_size(initial_batch_size)
                if current_batch_size != train_loader.batch_size:
                    train_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=current_batch_size,
                        shuffle=True
                    )
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Monitor resources
                    metrics = self.resource_manager.get_resource_usage()
                    
                    # Skip batch if resources are critically low
                    if metrics['ram_percent'] > 95 or (metrics['gpu_memory'] and metrics['gpu_memory'] > 95):
                        logging.warning("Resources critically low, skipping batch")
                        continue
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        logging.info(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                                   f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # Record metrics
                epoch_time = time.time() - epoch_start_time
                metrics = ResourceMetrics(
                    ram_usage=metrics['ram_percent'],
                    cpu_usage=metrics['cpu_percent'],
                    gpu_usage=metrics['gpu_usage'],
                    gpu_memory=metrics['gpu_memory'],
                    training_time=epoch_time,
                    batch_size=current_batch_size,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    performance_metrics={'loss': total_loss / len(train_loader)}
                )
                self.metrics_history.append(metrics)
                
                # Save checkpoint
                if (epoch + 1) % checkpoint_frequency == 0:
                    self.save_checkpoint(epoch, optimizer, total_loss / len(train_loader))
                
                # Generate resource report
                self.generate_resource_report(epoch)
                
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            # Try to save emergency checkpoint
            self.save_checkpoint(epoch, optimizer, total_loss / len(train_loader))
            raise

    def generate_resource_report(self, epoch: int):
        """Generate a report of resource usage and model performance."""
        try:
            report = {
                'epoch': epoch,
                'metrics': self.metrics_history[-1].__dict__,
                'resource_efficiency': {
                    'gpu_utilization': 'High' if self.metrics_history[-1].gpu_usage and 
                                               self.metrics_history[-1].gpu_usage > 80 else 'Low',
                    'memory_efficiency': 'Good' if self.metrics_history[-1].ram_usage < 80 else 'Poor',
                    'training_speed': f"{self.metrics_history[-1].training_time:.2f} seconds per epoch"
                },
                'recommendations': []
            }
            
            # Add recommendations based on metrics
            if self.metrics_history[-1].ram_usage > 80:
                report['recommendations'].append("Consider reducing batch size to improve memory usage")
            if self.metrics_history[-1].gpu_usage and self.metrics_history[-1].gpu_usage < 50:
                report['recommendations'].append("GPU underutilized - consider increasing batch size")
            
            # Save report
            report_path = self.checkpoint_dir / f'resource_report_epoch_{epoch}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logging.info(f"Generated resource report for epoch {epoch}")
            
        except Exception as e:
            logging.error(f"Error generating resource report: {str(e)}")

    def cleanup(self):
        """Clean up resources after training."""
        try:
            # Save final metrics
            metrics_path = self.checkpoint_dir / 'final_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump([m.__dict__ for m in self.metrics_history], f, indent=4)
            
            # Clear GPU memory if used
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            logging.info("Training cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
