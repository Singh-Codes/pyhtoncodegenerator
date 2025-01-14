"""
Main dependency management module that coordinates analysis, installation, and environment setup.
"""

import os
import logging
import json
from typing import Dict, List, Set, Optional, Any
from pathlib import Path

from .dependency_analyzer import DependencyAnalyzer
from .environment_manager import EnvironmentManager

class DependencyManager:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = DependencyAnalyzer()
        self.env_manager = EnvironmentManager(project_root)
        
        # Configure logging
        logging.basicConfig(
            filename='logs/dependency_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create logs directory if it doesn't exist
        self.project_root.joinpath('logs').mkdir(exist_ok=True)

    def setup_project_environment(self, python_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Set up a complete project environment with all dependencies.
        
        Args:
            python_version: Optional specific Python version to use
            
        Returns:
            Dict containing setup results and any issues encountered
        """
        results = {
            'status': 'success',
            'steps': [],
            'issues': []
        }
        
        try:
            # Step 1: Create virtual environment
            results['steps'].append('Creating virtual environment')
            if not self.env_manager.create_virtual_environment(python_version):
                results['status'] = 'error'
                results['issues'].append('Failed to create virtual environment')
                return results
            
            # Step 2: Analyze project dependencies
            results['steps'].append('Analyzing dependencies')
            dependencies = self.analyzer.analyze_directory(str(self.project_root))
            
            # Step 3: Check existing requirements.txt
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                existing_requirements = self.analyzer.parse_requirements_file(str(requirements_file))
                # Merge with found dependencies
                for package in dependencies['required']:
                    if package not in existing_requirements:
                        existing_requirements[package] = None
            else:
                existing_requirements = {pkg: None for pkg in dependencies['required']}
            
            # Step 4: Create backup of current environment
            results['steps'].append('Creating environment backup')
            backup_file = self.env_manager.backup_environment()
            if not backup_file:
                results['issues'].append('Warning: Failed to create environment backup')
            
            # Step 5: Install required dependencies
            results['steps'].append('Installing dependencies')
            installation_results = {}
            
            for package, version in existing_requirements.items():
                success, error = self.env_manager.install_package(package, version)
                if not success:
                    installation_results[package] = {
                        'status': 'error',
                        'error': error
                    }
                    results['issues'].append(f'Failed to install {package}: {error}')
                else:
                    installation_results[package] = {
                        'status': 'success'
                    }
            
            # Step 6: Install optional dependencies
            results['steps'].append('Installing optional dependencies')
            for package in dependencies['optional']:
                if package not in existing_requirements:
                    success, error = self.env_manager.install_package(package)
                    installation_results[package] = {
                        'status': 'success' if success else 'error',
                        'error': error if not success else None,
                        'optional': True
                    }
            
            # Step 7: Generate updated requirements.txt
            results['steps'].append('Updating requirements.txt')
            installed_packages = self.env_manager.get_installed_packages()
            self.analyzer.generate_requirements_file(installed_packages, str(requirements_file))
            
            # Add results
            results['installed_packages'] = installation_results
            results['requirements_file'] = str(requirements_file)
            if backup_file:
                results['backup_file'] = backup_file
            
            return results
            
        except Exception as e:
            logging.error(f"Error setting up project environment: {str(e)}")
            results['status'] = 'error'
            results['issues'].append(str(e))
            return results

    def check_project_health(self) -> Dict[str, Any]:
        """
        Check the health of project dependencies.
        
        Returns:
            Dict containing health check results
        """
        health_report = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check 1: Analyze current dependencies
            dependencies = self.analyzer.analyze_directory(str(self.project_root))
            installed_packages = self.env_manager.get_installed_packages()
            
            # Check for missing required packages
            missing_required = dependencies['required'] - set(installed_packages.keys())
            if missing_required:
                health_report['status'] = 'unhealthy'
                health_report['issues'].append({
                    'type': 'missing_required',
                    'packages': list(missing_required)
                })
            
            # Check for unused packages
            all_needed = dependencies['required'] | dependencies['optional']
            unused_packages = set(installed_packages.keys()) - all_needed
            if unused_packages:
                health_report['recommendations'].append({
                    'type': 'unused_packages',
                    'packages': list(unused_packages),
                    'message': 'Consider removing these unused packages'
                })
            
            # Check for version conflicts
            constraints = self.analyzer.analyze_version_constraints(all_needed)
            for package, requirements in constraints.items():
                if requirements:  # Has dependencies
                    for req in requirements:
                        if not self.analyzer.check_compatibility(package, installed_packages.get(package, '')):
                            health_report['issues'].append({
                                'type': 'version_conflict',
                                'package': package,
                                'requirement': req
                            })
            
            # Check requirements.txt health
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                file_requirements = self.analyzer.parse_requirements_file(str(requirements_file))
                
                # Check for outdated versions
                for package, version in file_requirements.items():
                    if version and package in installed_packages:
                        if version != installed_packages[package]:
                            health_report['issues'].append({
                                'type': 'version_mismatch',
                                'package': package,
                                'required': version,
                                'installed': installed_packages[package]
                            })
            else:
                health_report['recommendations'].append({
                    'type': 'missing_requirements',
                    'message': 'Consider creating a requirements.txt file'
                })
            
            return health_report
            
        except Exception as e:
            logging.error(f"Error checking project health: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def update_dependencies(self, packages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Update project dependencies.
        
        Args:
            packages: Optional list of specific packages to update
            
        Returns:
            Dict containing update results
        """
        results = {
            'status': 'success',
            'updated': [],
            'failed': [],
            'skipped': []
        }
        
        try:
            # Create backup before updating
            backup_file = self.env_manager.backup_environment()
            if not backup_file:
                results['warnings'] = ['Failed to create backup before updating']
            
            installed = self.env_manager.get_installed_packages()
            to_update = packages if packages else list(installed.keys())
            
            for package in to_update:
                if package in installed:
                    success, error = self.env_manager.install_package(package, upgrade=True)
                    if success:
                        results['updated'].append(package)
                    else:
                        results['failed'].append({
                            'package': package,
                            'error': error
                        })
                else:
                    results['skipped'].append(package)
            
            # Update requirements.txt
            if results['updated']:
                installed = self.env_manager.get_installed_packages()
                requirements_file = self.project_root / 'requirements.txt'
                self.analyzer.generate_requirements_file(installed, str(requirements_file))
            
            return results
            
        except Exception as e:
            logging.error(f"Error updating dependencies: {str(e)}")
            if backup_file:
                # Try to restore from backup
                self.env_manager.restore_environment(backup_file)
            return {
                'status': 'error',
                'error': str(e)
            }

    def generate_dependency_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report about project dependencies."""
        report = {
            'timestamp': self.analyzer.get_timestamp(),
            'project_root': str(self.project_root),
            'dependencies': {},
            'environment': {},
            'health_check': {},
            'recommendations': []
        }
        
        try:
            # Get dependency information
            deps = self.analyzer.analyze_directory(str(self.project_root))
            report['dependencies'] = {
                'required': list(deps['required']),
                'optional': list(deps['optional'])
            }
            
            # Get environment information
            installed = self.env_manager.get_installed_packages()
            report['environment'] = {
                'installed_packages': installed,
                'python_version': self.env_manager.get_python_version(),
                'virtual_env': str(self.env_manager.venv_path)
            }
            
            # Get health check
            health = self.check_project_health()
            report['health_check'] = health
            
            # Generate recommendations
            if health['status'] != 'healthy':
                for issue in health['issues']:
                    if issue['type'] == 'missing_required':
                        report['recommendations'].append({
                            'priority': 'high',
                            'message': f"Install missing required packages: {', '.join(issue['packages'])}"
                        })
                    elif issue['type'] == 'version_conflict':
                        report['recommendations'].append({
                            'priority': 'medium',
                            'message': f"Resolve version conflict for {issue['package']}"
                        })
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating dependency report: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
