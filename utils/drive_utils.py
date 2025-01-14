"""
Utility functions for Google Drive operations.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict
from google.colab import drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class DriveManager:
    """Manages Google Drive operations for training."""
    
    def __init__(self, root_folder: str = 'code_generator_ai'):
        self.root_folder = root_folder
        self.drive = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            filename='logs/drive_operations.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def mount_drive(self) -> bool:
        """Mount Google Drive."""
        try:
            drive.mount('/content/drive')
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()
            self.drive = GoogleDrive(gauth)
            logging.info("Successfully mounted Google Drive")
            return True
        except Exception as e:
            logging.error(f"Error mounting drive: {str(e)}")
            return False

    def create_folder_structure(self) -> bool:
        """Create necessary folder structure in Drive."""
        try:
            root = Path('/content/drive/MyDrive') / self.root_folder
            folders = ['checkpoints', 'data', 'logs', 'models']
            
            for folder in folders:
                folder_path = root / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                
            logging.info("Created folder structure")
            return True
        except Exception as e:
            logging.error(f"Error creating folders: {str(e)}")
            return False

    def save_checkpoint(self, 
                       checkpoint_data: Dict,
                       epoch: int,
                       max_checkpoints: int = 5) -> bool:
        """Save a model checkpoint to Drive."""
        try:
            checkpoint_dir = Path('/content/drive/MyDrive') / self.root_folder / 'checkpoints'
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir, max_checkpoints)
            
            logging.info(f"Saved checkpoint for epoch {epoch}")
            return True
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            return False

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, max_keep: int):
        """Remove old checkpoints, keeping only the most recent ones."""
        try:
            checkpoints = sorted(
                checkpoint_dir.glob('checkpoint_epoch_*.pt'),
                key=lambda x: int(x.stem.split('_')[-1])
            )
            
            # Remove old checkpoints
            while len(checkpoints) > max_keep:
                oldest = checkpoints.pop(0)
                oldest.unlink()
                logging.info(f"Removed old checkpoint: {oldest.name}")
                
        except Exception as e:
            logging.error(f"Error cleaning up checkpoints: {str(e)}")

    def load_checkpoint(self, epoch: Optional[int] = None) -> Optional[Dict]:
        """Load a checkpoint from Drive."""
        try:
            checkpoint_dir = Path('/content/drive/MyDrive') / self.root_folder / 'checkpoints'
            
            if epoch is None:
                # Load latest checkpoint
                checkpoints = sorted(
                    checkpoint_dir.glob('checkpoint_epoch_*.pt'),
                    key=lambda x: int(x.stem.split('_')[-1])
                )
                if not checkpoints:
                    return None
                checkpoint_path = checkpoints[-1]
            else:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                if not checkpoint_path.exists():
                    return None
            
            checkpoint = torch.load(checkpoint_path)
            logging.info(f"Loaded checkpoint: {checkpoint_path.name}")
            return checkpoint
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return None

    def save_logs(self, log_data: Dict, name: str) -> bool:
        """Save training logs to Drive."""
        try:
            log_dir = Path('/content/drive/MyDrive') / self.root_folder / 'logs'
            log_path = log_dir / f'{name}.json'
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=4)
                
            logging.info(f"Saved logs: {name}")
            return True
        except Exception as e:
            logging.error(f"Error saving logs: {str(e)}")
            return False

    def check_space(self) -> Dict[str, float]:
        """Check available space in Drive."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/content/drive")
            
            metrics = {
                'total_gb': total / (2**30),
                'used_gb': used / (2**30),
                'free_gb': free / (2**30),
                'usage_percent': (used / total) * 100
            }
            
            if metrics['usage_percent'] > 90:
                logging.warning("Drive storage nearly full!")
                
            return metrics
            
        except Exception as e:
            logging.error(f"Error checking space: {str(e)}")
            return {}

    def verify_file_upload(self, file_path: Path, max_retries: int = 3) -> bool:
        """Verify file was uploaded successfully."""
        for attempt in range(max_retries):
            try:
                if file_path.exists():
                    # Try to read file to verify
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # Read first 1KB
                    return True
                    
                time.sleep(1)  # Wait before retry
                
            except Exception as e:
                logging.warning(f"Verification attempt {attempt + 1} failed: {str(e)}")
                
        logging.error(f"Failed to verify upload: {file_path}")
        return False

    def cleanup(self):
        """Clean up temporary files and compress logs."""
        try:
            # Compress old logs
            log_dir = Path('/content/drive/MyDrive') / self.root_folder / 'logs'
            logs = list(log_dir.glob('*.json'))
            
            if len(logs) > 10:  # If more than 10 log files
                import zipfile
                from datetime import datetime
                
                # Create archive
                archive_name = f'logs_archive_{datetime.now().strftime("%Y%m%d")}.zip'
                archive_path = log_dir / archive_name
                
                with zipfile.ZipFile(archive_path, 'w') as zipf:
                    for log in logs[:-10]:  # Keep last 10 logs uncompressed
                        zipf.write(log, log.name)
                        log.unlink()  # Remove original file
                        
            logging.info("Cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
