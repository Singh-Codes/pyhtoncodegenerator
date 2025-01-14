# Google Drive Setup Guide for Code Generator AI

## Step 1: Create Project Folders in Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Create a new folder called `code_generator_ai`
3. Inside `code_generator_ai`, create these subfolders:
   - `checkpoints` - For model checkpoints
   - `data` - For training data
   - `logs` - For training logs
   - `models` - For saved models

## Step 2: Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project:
   - Click "Select a project" at the top
   - Click "New Project"
   - Name it "code-generator-ai"
   - Click "Create"

3. Enable required APIs:
   - Go to "APIs & Services" → "Library"
   - Search and enable these APIs:
     - Google Drive API
     - Cloud Storage API
     - Cloud Resource Manager API

4. Create credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "Service Account"
   - Name: "code-generator-training"
   - Click "Create"
   - Role: "Editor"
   - Click "Continue" → "Done"

5. Download credentials:
   - Click the service account email
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose JSON format
   - Save the file as `google_cloud_credentials.json`

## Step 3: Configure Local Environment

1. Place credentials file:
```bash
mkdir -p ~/.google/credentials
mv google_cloud_credentials.json ~/.google/credentials/
```

2. Set environment variables:
```bash
# Add to your .bashrc or .zshrc
export GOOGLE_APPLICATION_CREDENTIALS="~/.google/credentials/google_cloud_credentials.json"
export GOOGLE_CLOUD_PROJECT="code-generator-ai"
```

## Step 4: Configure Training Settings

1. Create a configuration file:
```python
# config/drive_config.py

DRIVE_CONFIG = {
    'root_folder': 'code_generator_ai',
    'checkpoints_folder': 'checkpoints',
    'data_folder': 'data',
    'logs_folder': 'logs',
    'models_folder': 'models',
    'checkpoint_frequency': 1,  # Save every epoch
    'max_checkpoints': 5,      # Keep last 5 checkpoints
}
```

## Step 5: Test Google Drive Connection

Run this test script:
```python
from google.colab import drive
from pathlib import Path
import json

def test_drive_setup():
    try:
        # Mount Drive
        drive.mount('/content/drive')
        
        # Check folders
        root = Path('/content/drive/MyDrive/code_generator_ai')
        required_folders = ['checkpoints', 'data', 'logs', 'models']
        
        for folder in required_folders:
            folder_path = root / folder
            if not folder_path.exists():
                print(f"Creating {folder}...")
                folder_path.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = root / 'test.txt'
        test_file.write_text('Drive setup successful!')
        
        # Test read access
        content = test_file.read_text()
        test_file.unlink()  # Clean up
        
        print("✅ Google Drive setup successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == '__main__':
    test_drive_setup()
```

## Step 6: Folder Structure

After setup, your Google Drive should have this structure:
```
code_generator_ai/
├── checkpoints/
│   └── checkpoint_epoch_[N].pt
├── data/
│   ├── raw/
│   ├── processed/
│   └── augmented/
├── logs/
│   ├── training_logs.txt
│   └── resource_metrics.json
└── models/
    ├── best_model.pt
    └── final_model.pt
```

## Step 7: Usage in Training

When training, use these paths:
```python
from pathlib import Path

# In Colab
DRIVE_ROOT = Path('/content/drive/MyDrive/code_generator_ai')
CHECKPOINTS_DIR = DRIVE_ROOT / 'checkpoints'
DATA_DIR = DRIVE_ROOT / 'data'
LOGS_DIR = DRIVE_ROOT / 'logs'
MODELS_DIR = DRIVE_ROOT / 'models'

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, CHECKPOINTS_DIR / f'checkpoint_epoch_{epoch}.pt')

# Load checkpoint
checkpoint = torch.load(CHECKPOINTS_DIR / 'checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Step 8: Best Practices

1. **Regular Cleanup**:
   - Delete old checkpoints (keep last 5)
   - Remove temporary files
   - Compress logs regularly

2. **Monitoring**:
   - Check Drive space regularly
   - Monitor upload/download speeds
   - Keep track of quota usage

3. **Security**:
   - Never share credential files
   - Use service accounts with minimal permissions
   - Regularly rotate credentials

4. **Error Recovery**:
   - Always verify file uploads
   - Keep local backups of critical files
   - Implement retry logic for Drive operations

## Common Issues and Solutions

1. **Drive Mount Fails**:
   ```python
   # Retry with force remount
   drive.mount('/content/drive', force_remount=True)
   ```

2. **Permission Denied**:
   - Check service account permissions
   - Verify folder sharing settings
   - Ensure credential file is correct

3. **Space Issues**:
   ```python
   # Check available space
   import shutil
   total, used, free = shutil.disk_usage("/content/drive")
   print(f"Free: {free // (2**30)} GiB")
   ```

4. **Slow Uploads**:
   - Use chunked uploads for large files
   - Compress data before upload
   - Use async operations for multiple files
