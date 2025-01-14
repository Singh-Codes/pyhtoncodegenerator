"""
Static code analysis module for identifying Python dependencies.
"""

import ast
import os
import re
import logging
import pkg_resources
from typing import Dict, List, Set, Tuple
from pathlib import Path

class DependencyAnalyzer:
    def __init__(self):
        self.builtin_modules = set(pkg_resources.working_set.by_key.keys())
        self.python_stdlib = self._get_stdlib_modules()
        
        # Configure logging
        logging.basicConfig(
            filename='logs/dependency_analyzer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _get_stdlib_modules(self) -> Set[str]:
        """Get a set of Python standard library module names."""
        import sys
        import distutils.sysconfig as sysconfig
        
        stdlib_path = sysconfig.get_python_lib(standard_lib=True)
        stdlib_modules = set()
        
        # Add built-in modules
        stdlib_modules.update(sys.builtin_module_names)
        
        # Add modules from standard library path
        for path in Path(stdlib_path).rglob('*.py'):
            module_name = path.stem
            if module_name != '__init__':
                stdlib_modules.add(module_name)
        
        return stdlib_modules

    def analyze_file(self, file_path: str) -> Dict[str, List[str]]:
        """
        Analyze a Python file for dependencies.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dict containing direct imports and their requirements
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = {
                'direct': set(),  # Direct imports
                'from': set(),    # From imports
                'conditional': set()  # Imports inside try/except blocks
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports['direct'].add(name.name.split('.')[0])
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports['from'].add(node.module.split('.')[0])
                        
                elif isinstance(node, ast.Try):
                    # Check for imports in try blocks (optional dependencies)
                    for handler in node.handlers:
                        for n in ast.walk(handler):
                            if isinstance(n, (ast.Import, ast.ImportFrom)):
                                if isinstance(n, ast.Import):
                                    for name in n.names:
                                        imports['conditional'].add(name.name.split('.')[0])
                                else:
                                    if n.module:
                                        imports['conditional'].add(n.module.split('.')[0])
            
            # Filter out standard library modules
            for import_type in imports:
                imports[import_type] = {
                    module for module in imports[import_type]
                    if module not in self.python_stdlib
                }
            
            return imports
            
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {str(e)}")
            return {'direct': set(), 'from': set(), 'conditional': set()}

    def analyze_directory(self, directory: str) -> Dict[str, Set[str]]:
        """
        Recursively analyze all Python files in a directory.
        
        Args:
            directory: Path to the directory
            
        Returns:
            Dict containing all dependencies found
        """
        all_imports = {
            'required': set(),  # Must-have dependencies
            'optional': set()   # Optional dependencies
        }
        
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        imports = self.analyze_file(file_path)
                        
                        # Add direct and from imports to required
                        all_imports['required'].update(imports['direct'])
                        all_imports['required'].update(imports['from'])
                        
                        # Add conditional imports to optional
                        all_imports['optional'].update(imports['conditional'])
            
            return all_imports
            
        except Exception as e:
            logging.error(f"Error analyzing directory {directory}: {str(e)}")
            return {'required': set(), 'optional': set()}

    def get_installed_versions(self, packages: Set[str]) -> Dict[str, str]:
        """Get installed versions of packages."""
        versions = {}
        for package in packages:
            try:
                dist = pkg_resources.get_distribution(package)
                versions[package] = dist.version
            except pkg_resources.DistributionNotFound:
                versions[package] = None
        return versions

    def check_compatibility(self, package: str, version: str) -> bool:
        """Check if a package version is compatible with the current Python version."""
        try:
            pkg_resources.require(f"{package}=={version}")
            return True
        except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
            return False

    def parse_requirements_file(self, file_path: str) -> Dict[str, str]:
        """Parse a requirements.txt file."""
        requirements = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle different requirement formats
                        if '==' in line:
                            package, version = line.split('==')
                            requirements[package] = version
                        elif '>=' in line:
                            package, version = line.split('>=')
                            requirements[package] = f">={version}"
                        else:
                            requirements[line] = None
        return requirements

    def generate_requirements_file(self, packages: Dict[str, str], file_path: str):
        """Generate a requirements.txt file."""
        try:
            with open(file_path, 'w') as f:
                for package, version in sorted(packages.items()):
                    if version:
                        f.write(f"{package}=={version}\n")
                    else:
                        f.write(f"{package}\n")
        except Exception as e:
            logging.error(f"Error generating requirements file: {str(e)}")

    def analyze_version_constraints(self, packages: Set[str]) -> Dict[str, List[str]]:
        """Analyze version constraints and conflicts between packages."""
        constraints = {}
        for package in packages:
            try:
                dist = pkg_resources.get_distribution(package)
                requires = dist.requires()
                constraints[package] = [str(req) for req in requires]
            except pkg_resources.DistributionNotFound:
                constraints[package] = []
        return constraints
