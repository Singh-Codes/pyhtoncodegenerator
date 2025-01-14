"""
Helper script to set up Google Colab environment.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict
import logging

def setup_colab_environment():
    """Set up the Colab environment with required files and structure."""
    try:
        # Mount Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create directories
        colab_root = Path('/content/code_generator_ai')
        drive_root = Path('/content/drive/MyDrive/code_generator_ai')
        
        # Create Colab directories
        directories = [
            'model',
            'training',
            'utils',
            'error_resolution',
            'dependency_manager'
        ]
        
        for dir_name in directories:
            (colab_root / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create Drive directories
        drive_directories = [
            'checkpoints',
            'data',
            'logs',
            'models'
        ]
        
        for dir_name in drive_directories:
            (drive_root / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Add project root to Python path
        if str(colab_root) not in sys.path:
            sys.path.insert(0, str(colab_root))
        
        # Fix permissions
        os.system(f'chmod -R 755 {colab_root}')
        
        print("✅ Colab environment setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Colab environment: {str(e)}")
        return False

def verify_files(required_files: List[str]) -> List[str]:
    """Verify all required files are present."""
    colab_root = Path('/content/code_generator_ai')
    missing_files = []
    
    for file in required_files:
        if not (colab_root / file).exists():
            missing_files.append(file)
    
    return missing_files

def setup_logging():
    """Configure logging for Colab training."""
    log_dir = Path('/content/drive/MyDrive/code_generator_ai/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / 'colab_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_imports():
    """Test that all required imports work."""
    try:
        from model.transformer import TransformerModel
        from training.colab_trainer import ColabTrainer
        from training.data_pipeline import DataPipeline
        from utils.drive_utils import DriveManager
        
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def main():
    """Main setup function."""
    # Required files to check
    required_files = [
        'model/transformer.py',
        'training/colab_trainer.py',
        'training/code_generation_trainer.py',
        'training/data_pipeline.py',
        'utils/drive_utils.py',
        'error_resolution/error_detector.py',
        'error_resolution/web_scraper.py',
        'error_resolution/nlp_analyzer.py',
        'error_resolution/solution_manager.py',
        'dependency_manager/dependency_analyzer.py',
        'dependency_manager/environment_manager.py',
        'requirements.txt',
        'train.py'
    ]
    
    # Setup environment
    if not setup_colab_environment():
        return
    
    # Verify files
    missing = verify_files(required_files)
    if missing:
        print("\n❌ Missing files:")
        for file in missing:
            print(f"  - {file}")
        return
    
    # Setup logging
    setup_logging()
    
    # Test imports
    if not test_imports():
        return
    
    print("\n✅ Colab setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: !python train.py --mode colab")
    print("2. Monitor logs: !tail -f /content/drive/MyDrive/code_generator_ai/logs/colab_training.log")
    print("3. Check GPU: !nvidia-smi")

if __name__ == '__main__':
    main()
