"""
Verify that all required dependencies are installed and working.
"""

def check_dependencies():
    """Check if all required dependencies are installed and working."""
    dependencies = {}
    
    # Core Dependencies
    print("\n=== Checking Core Dependencies ===")
    try:
        import torch
        import transformers
        import datasets
        import numpy as np
        import pandas as pd
        dependencies.update({
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'datasets': datasets.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__
        })
        print("[PASS] Core dependencies installed")
    except ImportError as e:
        print(f"[FAIL] Error in core dependencies: {str(e)}")
    
    # Training Tools
    print("\n=== Checking Training Tools ===")
    try:
        import wandb
        import tensorboard
        import tqdm
        import sklearn
        dependencies.update({
            'wandb': wandb.__version__,
            'tensorboard': tensorboard.__version__,
            'tqdm': tqdm.__version__,
            'scikit-learn': sklearn.__version__
        })
        print("[PASS] Training tools installed")
    except ImportError as e:
        print(f"[FAIL] Error in training tools: {str(e)}")
    
    # Code Processing
    print("\n=== Checking Code Processing Tools ===")
    try:
        import tokenizers
        import rouge_score
        import sacrebleu
        import astor
        dependencies.update({
            'tokenizers': tokenizers.__version__,
            'rouge_score': rouge_score.__version__,
            'sacrebleu': sacrebleu.__version__,
            'astor': astor.__version__
        })
        print("[PASS] Code processing tools installed")
    except ImportError as e:
        print(f"[FAIL] Error in code processing tools: {str(e)}")
    
    # Metrics and Monitoring
    print("\n=== Checking Metrics and Monitoring ===")
    try:
        import codecarbon
        import GPUtil
        import psutil
        import py_cpuinfo
        dependencies.update({
            'codecarbon': codecarbon.__version__,
            'GPUtil': GPUtil.__version__,
            'psutil': psutil.__version__,
            'py-cpuinfo': py_cpuinfo.__version__
        })
        print("[PASS] Metrics and monitoring tools installed")
    except ImportError as e:
        print(f"[FAIL] Error in metrics and monitoring: {str(e)}")
    
    # GPU Check
    print("\n=== Checking GPU Availability ===")
    if torch.cuda.is_available():
        print(f"[PASS] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("[INFO] No GPU found, will use CPU for training")
    
    # Print all versions
    print("\n=== Installed Versions ===")
    for package, version in dependencies.items():
        print(f"{package}: {version}")

if __name__ == "__main__":
    check_dependencies()
