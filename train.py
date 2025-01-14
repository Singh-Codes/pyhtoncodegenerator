"""
Main training script for the Code Generator AI.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from transformers import AutoTokenizer

from model.transformer import TransformerModel
from training.colab_trainer import ColabTrainer, ColabResourceManager
from training.data_pipeline import DataPipeline
from training.code_generation_trainer import CodeGenerationTrainer

def setup_argparse():
    parser = argparse.ArgumentParser(description='Train Code Generator AI')
    parser.add_argument('--mode', choices=['local', 'colab'], default='local',
                       help='Training mode: local or Google Colab')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing training data')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    return parser.parse_args()

def train_local(args):
    """Train the model locally."""
    # Initialize model and tokenizer
    model = TransformerModel()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Initialize trainer
    trainer = CodeGenerationTrainer(model, tokenizer)
    
    # Generate training scenarios
    scenarios = trainer.generate_training_scenarios()
    
    # Train the model
    trainer.train(
        train_scenarios=scenarios,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Save the final model
    model_path = Path(args.model_dir) / 'final_model'
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")

def train_colab(args):
    """Train the model on Google Colab."""
    # Initialize components
    resource_manager = ColabResourceManager()
    model = TransformerModel()
    trainer = ColabTrainer(model, resource_manager)
    
    # Set up data pipeline
    pipeline = DataPipeline(args.data_dir)
    
    # Define preprocessing steps
    preprocessing_steps = [
        {
            "type": "encode_categorical",
            "columns": ["language", "framework", "library"]
        },
        {
            "type": "normalize",
            "columns": ["code_length", "complexity_score"]
        }
    ]
    
    try:
        # Preprocess data
        print("Preprocessing data...")
        preprocessed_data = pipeline.preprocess_locally(
            os.path.join(args.data_dir, "training_data.csv"),
            preprocessing_steps
        )
        
        # Create data splits
        print("Creating data splits...")
        splits = pipeline.create_data_splits(preprocessed_data)
        
        # Data augmentation config
        augmentation_config = {
            "noise": {
                "columns": ["code_length", "complexity_score"],
                "std": 0.1
            },
            "shuffle": {
                "columns": ["language", "framework"]
            }
        }
        
        # Augment training data
        print("Augmenting training data...")
        augmented_train = pipeline.augment_data(
            splits['train'],
            augmentation_config
        )
        
        # Upload to cloud storage
        print("Uploading data to cloud storage...")
        pipeline.upload_to_cloud(
            "code-generator-bucket",  # Replace with your bucket name
            pipeline.local_data_dir / "augmented_data.csv",
            "training/augmented_data.csv"
        )
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader = pipeline.create_streaming_dataloader(
            "code-generator-bucket",  # Replace with your bucket name
            "training/augmented_data.csv",
            args.batch_size
        )
        
        # Initialize optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        print("Starting training...")
        trainer.train(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.epochs,
            initial_batch_size=args.batch_size,
            checkpoint_frequency=1
        )
        
        # Generate final report
        trainer.generate_resource_report(args.epochs - 1)
        
        # Cleanup
        print("Cleaning up...")
        trainer.cleanup()
        pipeline.cleanup()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Attempt to clean up
        trainer.cleanup()
        pipeline.cleanup()
        raise

def main():
    # Parse arguments
    args = setup_argparse()
    
    # Create necessary directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up logging directory
    Path('logs').mkdir(exist_ok=True)
    
    # Train based on mode
    if args.mode == 'local':
        print("Starting local training...")
        train_local(args)
    else:
        print("Starting Colab training...")
        train_colab(args)

if __name__ == '__main__':
    main()
