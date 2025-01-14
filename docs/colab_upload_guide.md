# Google Colab Upload Guide

## Required Files and Folders

Here's what you need to upload to Colab:

```
code_generator_ai/
├── model/
│   ├── __init__.py
│   └── transformer.py
├── training/
│   ├── __init__.py
│   ├── colab_trainer.py
│   ├── code_generation_trainer.py
│   └── data_pipeline.py
├── utils/
│   ├── __init__.py
│   └── drive_utils.py
├── error_resolution/
│   ├── __init__.py
│   ├── error_detector.py
│   ├── web_scraper.py
│   ├── nlp_analyzer.py
│   └── solution_manager.py
├── dependency_manager/
│   ├── __init__.py
│   ├── dependency_analyzer.py
│   └── environment_manager.py
├── requirements.txt
└── train.py
```

## Upload Instructions

### Method 1: Direct Upload to Colab

```python
# Run this in a Colab cell
!mkdir -p /content/code_generator_ai
```

Then use Colab's file upload button (folder icon on the left) to upload the files.

### Method 2: Upload via Google Drive

1. Create folder structure in Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Create directories
!mkdir -p /content/drive/MyDrive/code_generator_ai/{model,training,utils,error_resolution,dependency_manager}
```

2. Upload files to corresponding folders in Drive:
```python
# Copy from Drive to Colab
!cp -r /content/drive/MyDrive/code_generator_ai/* /content/code_generator_ai/
```

### Method 3: Git Repository (Recommended)

```python
# Clone the repository
!git clone https://github.com/your-username/code-generator-ai.git
!cd code_generator_ai
```

## Verify Upload

Run this verification script:
```python
def verify_upload():
    import os
    
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
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(f'/content/code_generator_ai/{file}'):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("✅ All required files present!")

verify_upload()
```

## Install Dependencies

After uploading, install the required packages:
```python
!pip install -r /content/code_generator_ai/requirements.txt
```

## Directory Structure in Colab

Your final Colab workspace should look like this:
```
/content/
├── code_generator_ai/          # Code files
│   ├── model/
│   ├── training/
│   ├── utils/
│   ├── error_resolution/
│   └── dependency_manager/
└── drive/
    └── MyDrive/
        └── code_generator_ai/  # Training data and outputs
            ├── checkpoints/
            ├── data/
            ├── logs/
            └── models/
```

## Important Notes

1. **Code Files vs Training Data**:
   - Code files go in `/content/code_generator_ai/`
   - Training data and outputs go in `/content/drive/MyDrive/code_generator_ai/`

2. **File Permissions**:
   ```python
   # Make sure files are executable
   !chmod +x /content/code_generator_ai/train.py
   ```

3. **Path Configuration**:
   ```python
   import sys
   sys.path.append('/content/code_generator_ai')
   ```

4. **Test Import**:
   ```python
   # Verify imports work
   from model.transformer import TransformerModel
   from training.colab_trainer import ColabTrainer
   print("✅ Imports successful!")
   ```

## Common Issues and Solutions

1. **Import Errors**:
   ```python
   # Add project root to Python path
   import sys
   sys.path.insert(0, '/content/code_generator_ai')
   ```

2. **Permission Issues**:
   ```python
   # Fix permissions
   !chmod -R 755 /content/code_generator_ai
   ```

3. **Missing Files**:
   ```python
   # Check file existence
   !ls -R /content/code_generator_ai
   ```

4. **Drive Mount Issues**:
   ```python
   # Force remount
   drive.mount('/content/drive', force_remount=True)
   ```
