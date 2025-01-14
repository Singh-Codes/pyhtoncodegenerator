"""
Dependency Management System for Python Code Generator AI.

This package provides comprehensive dependency management capabilities:
1. Static code analysis to detect required libraries
2. Virtual environment management
3. Automatic dependency installation and version management
4. Health checks and monitoring
5. Dependency reports and recommendations
"""

from .dependency_analyzer import DependencyAnalyzer
from .environment_manager import EnvironmentManager
from .dependency_manager import DependencyManager

__all__ = [
    'DependencyAnalyzer',
    'EnvironmentManager',
    'DependencyManager'
]
