"""
Virtual environment and package management module.
"""

import os
import sys
import venv
import subprocess
import logging
import platform
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class EnvironmentManager:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.venv_path = self.project_root / "venv"
        
        # Configure logging
        logging.basicConfig(
            filename='logs/environment_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Platform-specific settings
        self.is_windows = platform.system() == "Windows"
        self.python_executable = self._get_python_executable()
        self.pip_executable = self._get_pip_executable()

    def _get_python_executable(self) -> str:
        """Get the path to the Python executable in the virtual environment."""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")

    def _get_pip_executable(self) -> str:
        """Get the path to the pip executable in the virtual environment."""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "pip.exe")
        return str(self.venv_path / "bin" / "pip")

    def create_virtual_environment(self, python_version: Optional[str] = None) -> bool:
        """
        Create a new virtual environment.
        
        Args:
            python_version: Optional specific Python version to use
            
        Returns:
            bool indicating success
        """
        try:
            if self.venv_path.exists():
                logging.info(f"Virtual environment already exists at {self.venv_path}")
                return True
            
            # Create virtual environment
            builder = venv.EnvBuilder(
                system_site_packages=False,
                clear=True,
                with_pip=True,
                upgrade_deps=True
            )
            builder.create(self.venv_path)
            
            logging.info(f"Created virtual environment at {self.venv_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating virtual environment: {str(e)}")
            return False

    def install_package(self, package: str, version: Optional[str] = None, upgrade: bool = False) -> Tuple[bool, str]:
        """
        Install a Python package in the virtual environment.
        
        Args:
            package: Name of the package to install
            version: Optional specific version to install
            upgrade: Whether to upgrade an existing installation
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            cmd = [self.pip_executable, "install"]
            
            if upgrade:
                cmd.append("--upgrade")
            
            if version:
                cmd.append(f"{package}=={version}")
            else:
                cmd.append(package)
            
            # Run pip install
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully installed {package}")
                return True, ""
            else:
                error_msg = result.stderr
                logging.error(f"Error installing {package}: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Exception installing {package}: {error_msg}")
            return False, error_msg

    def install_requirements(self, requirements_file: str) -> Dict[str, str]:
        """
        Install packages from a requirements file.
        
        Returns:
            Dict mapping package names to installation status/error messages
        """
        results = {}
        
        try:
            cmd = [
                self.pip_executable,
                "install",
                "-r",
                requirements_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logging.info("Successfully installed all requirements")
                results["status"] = "success"
            else:
                logging.error(f"Error installing requirements: {result.stderr}")
                results["status"] = "error"
                results["error"] = result.stderr
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Exception installing requirements: {error_msg}")
            results["status"] = "error"
            results["error"] = error_msg
        
        return results

    def check_package_installed(self, package: str) -> bool:
        """Check if a package is installed in the virtual environment."""
        try:
            cmd = [
                self.pip_executable,
                "show",
                package
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"Error checking package {package}: {str(e)}")
            return False

    def get_installed_packages(self) -> Dict[str, str]:
        """Get a list of all installed packages and their versions."""
        try:
            cmd = [
                self.pip_executable,
                "list",
                "--format=json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                import json
                packages = json.loads(result.stdout)
                return {pkg["name"]: pkg["version"] for pkg in packages}
            
            return {}
            
        except Exception as e:
            logging.error(f"Error getting installed packages: {str(e)}")
            return {}

    def backup_environment(self) -> Optional[str]:
        """
        Create a backup of the current environment.
        
        Returns:
            Path to the backup requirements file
        """
        try:
            backup_file = self.project_root / "requirements.backup.txt"
            
            cmd = [
                self.pip_executable,
                "freeze",
                "--all"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                with open(backup_file, 'w') as f:
                    f.write(result.stdout)
                logging.info(f"Created environment backup at {backup_file}")
                return str(backup_file)
            
            return None
            
        except Exception as e:
            logging.error(f"Error backing up environment: {str(e)}")
            return None

    def restore_environment(self, backup_file: str) -> bool:
        """Restore the environment from a backup file."""
        try:
            if not os.path.exists(backup_file):
                logging.error(f"Backup file not found: {backup_file}")
                return False
            
            # First uninstall all packages
            cmd_uninstall = [
                self.pip_executable,
                "uninstall",
                "-y",
                "-r",
                backup_file
            ]
            
            subprocess.run(cmd_uninstall, check=False)
            
            # Then install from backup
            cmd_install = [
                self.pip_executable,
                "install",
                "-r",
                backup_file
            ]
            
            result = subprocess.run(
                cmd_install,
                capture_output=True,
                text=True,
                check=False
            )
            
            success = result.returncode == 0
            if success:
                logging.info("Successfully restored environment from backup")
            else:
                logging.error(f"Error restoring environment: {result.stderr}")
            
            return success
            
        except Exception as e:
            logging.error(f"Exception restoring environment: {str(e)}")
            return False
