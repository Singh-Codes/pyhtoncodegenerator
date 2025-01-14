"""
Script to automatically update project documentation.
"""

import os
import ast
from datetime import datetime
import time
from pathlib import Path
import subprocess

class DocumentationUpdater:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.doc_file = self.project_root / 'project_documentation.txt'
        self.components = {
            'model/transformer.py': self._analyze_transformer,
            'model/train.py': self._analyze_training,
            'model/inference.py': self._analyze_inference,
            'scripts/preprocess_data.py': self._analyze_preprocessing
        }

    def _get_git_changes(self) -> str:
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%s'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            return result.stdout
        except:
            return "No git history available"

    def _analyze_transformer(self, path: Path) -> dict:
        """Analyze transformer.py implementation status."""
        status = {
            'implemented': [],
            'pending': []
        }
        
        try:
            with open(path, 'r') as f:
                tree = ast.parse(f.read())
            
            # Check for key components
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name == 'CodeGeneratorTransformer':
                        status['implemented'].append('Transformer architecture')
                    if node.name == 'PositionalEncoding':
                        status['implemented'].append('Positional encoding')
                        
        except Exception as e:
            status['pending'].append(f'Error analyzing transformer: {str(e)}')
            
        return status

    def _analyze_training(self, path: Path) -> dict:
        """Analyze training implementation status."""
        status = {
            'implemented': [],
            'pending': []
        }
        
        try:
            with open(path, 'r') as f:
                content = f.read()
                
            if 'train_epoch' in content:
                status['implemented'].append('Training loop')
            if 'evaluate' in content:
                status['implemented'].append('Validation process')
            if 'save_model' in content:
                status['implemented'].append('Model checkpointing')
                
        except Exception as e:
            status['pending'].append(f'Error analyzing training: {str(e)}')
            
        return status

    def _analyze_preprocessing(self, path: Path) -> dict:
        """Analyze preprocessing implementation status."""
        status = {
            'implemented': [],
            'pending': []
        }
        
        try:
            with open(path, 'r') as f:
                content = f.read()
                
            if 'train_tokenizer' in content:
                status['implemented'].append('Tokenizer training')
            if 'process_file' in content:
                status['implemented'].append('File processing')
            if 'extract_docstring' in content:
                status['implemented'].append('Docstring extraction')
                
        except Exception as e:
            status['pending'].append(f'Error analyzing preprocessing: {str(e)}')
            
        return status

    def _analyze_inference(self, path: Path) -> dict:
        """Analyze inference implementation status."""
        status = {
            'implemented': [],
            'pending': []
        }
        
        try:
            with open(path, 'r') as f:
                content = f.read()
                
            if 'generate_code' in content:
                status['implemented'].append('Code generation')
            if 'clean_generated_code' in content:
                status['implemented'].append('Code cleanup')
                
        except Exception as e:
            status['pending'].append(f'Error analyzing inference: {str(e)}')
            
        return status

    def update_documentation(self):
        """Update the project documentation file."""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Analyze all components
        component_status = {}
        for file_path, analyzer in self.components.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                component_status[file_path] = analyzer(full_path)
        
        # Generate documentation content
        content = f"""Python Code Generator AI - Project Documentation
Last Updated: {current_time}

1. Project Overview
------------------
A custom AI-powered Python code generator that creates code from natural language descriptions.
Built using a transformer-based architecture without relying on pre-trained models.

2. Current Status
----------------
"""
        
        # Add component status
        content += "Components Implementation Status:\n"
        for component, status in component_status.items():
            content += f"\n{component}:\n"
            for impl in status['implemented']:
                content += f"[+] {impl}\n"
            for pending in status['pending']:
                content += f"[-] Pending: {pending}\n"
        
        # Add recent changes
        recent_changes = self._get_git_changes()
        content += f"\n3. Recent Changes\n----------------\n{recent_changes}\n"
        
        # Add known issues
        content += "\n4. Known Issues\n--------------\n"
        content += "- PyTorch installation pending\n"
        content += "- Initial model training pending\n"
        
        # Write to file with UTF-8 encoding
        with open(self.doc_file, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    project_root = Path(__file__).parent.parent
    updater = DocumentationUpdater(project_root)
    
    while True:
        print(f"Updating documentation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        updater.update_documentation()
        time.sleep(300)  # Update every 5 minutes

if __name__ == '__main__':
    main()
