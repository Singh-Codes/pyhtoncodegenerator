import os
import zipfile

def create_project_zip():
    """Create a ZIP file of the project for Colab."""
    project_files = [
        'model/transformer.py',
        'model/__init__.py',
        'utils/augmentation.py',
        'utils/metrics.py',
        'examples/code_examples.py',
        'requirements.txt',
        'code_generator_training.ipynb'
    ]
    
    with zipfile.ZipFile('code_generator_ai.zip', 'w') as zipf:
        for file in project_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file} to ZIP")
            else:
                print(f"Warning: {file} not found")

if __name__ == "__main__":
    create_project_zip()
