"""
Data pipeline for efficient data handling and preprocessing.
"""

import os
import io
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split

class StreamingDataset(IterableDataset):
    """Dataset that streams data directly from cloud storage."""
    
    def __init__(self, 
                 bucket_name: str,
                 data_path: str,
                 chunk_size: int = 1000,
                 transform=None):
        self.bucket_name = bucket_name
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.transform = transform
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        
    def __iter__(self):
        blob = self.bucket.get_blob(self.data_path)
        stream = io.BytesIO(blob.download_as_bytes())
        
        while True:
            chunk = pd.read_csv(stream, nrows=self.chunk_size)
            if chunk.empty:
                break
                
            for _, row in chunk.iterrows():
                data = self.transform(row) if self.transform else row
                yield data

class DataPipeline:
    """Manages data preprocessing and streaming."""
    
    def __init__(self, local_data_dir: str):
        self.local_data_dir = Path(local_data_dir)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for data pipeline."""
        logging.basicConfig(
            filename='logs/data_pipeline.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def preprocess_locally(self, 
                          data_path: str,
                          preprocessing_steps: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess data locally before upload."""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Apply preprocessing steps
            for step in preprocessing_steps:
                if step['type'] == 'drop_columns':
                    df = df.drop(columns=step['columns'])
                elif step['type'] == 'fill_na':
                    df = df.fillna(step['value'])
                elif step['type'] == 'normalize':
                    for col in step['columns']:
                        df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif step['type'] == 'encode_categorical':
                    for col in step['columns']:
                        df[col] = pd.Categorical(df[col]).codes
                elif step['type'] == 'custom':
                    df = step['function'](df)
            
            # Save preprocessed data
            preprocessed_path = self.local_data_dir / 'preprocessed_data.csv'
            df.to_csv(preprocessed_path, index=False)
            
            logging.info(f"Preprocessing completed, saved to {preprocessed_path}")
            return df
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")
            raise

    def create_data_splits(self, 
                          df: pd.DataFrame,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits."""
        try:
            # First split into train+val and test
            train_val, test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )
            
            # Then split train+val into train and val
            val_ratio = val_size / (1 - test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio,
                random_state=random_state
            )
            
            splits = {
                'train': train,
                'val': val,
                'test': test
            }
            
            # Save splits locally
            for split_name, split_data in splits.items():
                split_path = self.local_data_dir / f'{split_name}_data.csv'
                split_data.to_csv(split_path, index=False)
            
            logging.info("Data splits created and saved")
            return splits
            
        except Exception as e:
            logging.error(f"Error creating data splits: {str(e)}")
            raise

    def upload_to_cloud(self, 
                       bucket_name: str,
                       local_path: Union[str, Path],
                       cloud_path: str):
        """Upload data to cloud storage."""
        try:
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(cloud_path)
            blob.upload_from_filename(str(local_path))
            
            logging.info(f"Uploaded {local_path} to {cloud_path}")
            
        except Exception as e:
            logging.error(f"Error uploading to cloud: {str(e)}")
            raise

    def create_streaming_dataloader(self,
                                  bucket_name: str,
                                  data_path: str,
                                  batch_size: int,
                                  transform=None) -> DataLoader:
        """Create a DataLoader for streaming data."""
        try:
            dataset = StreamingDataset(
                bucket_name=bucket_name,
                data_path=data_path,
                transform=transform
            )
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2
            )
            
        except Exception as e:
            logging.error(f"Error creating streaming dataloader: {str(e)}")
            raise

    def augment_data(self, 
                    df: pd.DataFrame,
                    augmentation_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply data augmentation techniques."""
        try:
            augmented_data = []
            
            for _, row in df.iterrows():
                # Add original data
                augmented_data.append(row)
                
                # Apply augmentations
                for aug_type, params in augmentation_config.items():
                    if aug_type == 'noise':
                        # Add random noise
                        noisy_row = row.copy()
                        for col in params['columns']:
                            noise = np.random.normal(0, params['std'], 1)[0]
                            noisy_row[col] += noise
                        augmented_data.append(noisy_row)
                        
                    elif aug_type == 'shuffle':
                        # Shuffle specified columns
                        shuffled_row = row.copy()
                        shuffle_cols = params['columns']
                        shuffle_values = shuffled_row[shuffle_cols].values
                        np.random.shuffle(shuffle_values)
                        shuffled_row[shuffle_cols] = shuffle_values
                        augmented_data.append(shuffled_row)
                        
                    elif aug_type == 'custom':
                        # Apply custom augmentation function
                        augmented_row = params['function'](row)
                        augmented_data.append(augmented_row)
            
            augmented_df = pd.DataFrame(augmented_data)
            
            # Save augmented data
            augmented_path = self.local_data_dir / 'augmented_data.csv'
            augmented_df.to_csv(augmented_path, index=False)
            
            logging.info(f"Data augmentation completed, saved to {augmented_path}")
            return augmented_df
            
        except Exception as e:
            logging.error(f"Error during data augmentation: {str(e)}")
            raise

    def monitor_data_pipeline(self) -> Dict[str, Any]:
        """Monitor data pipeline performance and storage usage."""
        try:
            metrics = {
                'local_storage': {
                    'total_size': 0,
                    'files': {}
                },
                'processing_stats': {
                    'num_files': 0,
                    'total_rows': 0,
                    'memory_usage': 0
                }
            }
            
            # Calculate local storage usage
            for file_path in self.local_data_dir.glob('*.csv'):
                file_size = file_path.stat().st_size
                metrics['local_storage']['files'][file_path.name] = file_size
                metrics['local_storage']['total_size'] += file_size
                
                # Get file statistics
                df = pd.read_csv(file_path)
                metrics['processing_stats']['num_files'] += 1
                metrics['processing_stats']['total_rows'] += len(df)
                metrics['processing_stats']['memory_usage'] += df.memory_usage(deep=True).sum()
            
            # Save metrics
            metrics_path = self.local_data_dir / 'pipeline_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logging.info("Data pipeline metrics generated")
            return metrics
            
        except Exception as e:
            logging.error(f"Error monitoring data pipeline: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            # Remove temporary files but keep preprocessed data
            for file_path in self.local_data_dir.glob('temp_*.csv'):
                file_path.unlink()
            
            logging.info("Data pipeline cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            raise
