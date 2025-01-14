"""
Training module for the Code Generator AI.

This package provides training capabilities for various coding scenarios:
1. Basic function generation
2. Error handling
3. API integration
4. Database operations
5. File handling
6. Dependency management
7. Web development
8. Data processing
"""

from .code_generation_trainer import CodeGenerationTrainer, CodeGenerationDataset

__all__ = [
    'CodeGenerationTrainer',
    'CodeGenerationDataset'
]
