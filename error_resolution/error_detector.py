"""
Error detection and logging module for the AI code generator.
"""

import sys
import traceback
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ErrorDetector:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            filename=self.log_dir / "error_detector.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Error database file
        self.error_db_file = self.log_dir / "error_database.json"
        self.error_database = self._load_error_database()

    def _load_error_database(self) -> Dict[str, Any]:
        """Load the error database from file."""
        if self.error_db_file.exists():
            try:
                with open(self.error_db_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning("Error database corrupted, creating new one")
                return {"errors": {}}
        return {"errors": {}}

    def _save_error_database(self):
        """Save the error database to file."""
        with open(self.error_db_file, 'w') as f:
            json.dump(self.error_database, f, indent=2)

    def get_error_signature(self, error: Exception, code_context: str) -> str:
        """Generate a unique signature for an error based on its type and context."""
        tb = traceback.extract_tb(sys.exc_info()[2])
        error_location = tb[-1] if tb else None
        
        signature_parts = [
            error.__class__.__name__,
            str(error),
            error_location.filename if error_location else "",
            str(error_location.lineno) if error_location else "",
        ]
        return "|".join(signature_parts)

    def log_error(self, error: Exception, code_context: str) -> Dict[str, Any]:
        """
        Log an error and its context to the database.
        
        Args:
            error: The exception that occurred
            code_context: The code that caused the error
            
        Returns:
            Dict containing error details
        """
        error_signature = self.get_error_signature(error, code_context)
        
        error_info = {
            "type": error.__class__.__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
            "code_context": code_context,
            "resolved": False,
            "solutions_attempted": [],
            "successful_solution": None
        }
        
        # Update database
        if error_signature not in self.error_database["errors"]:
            self.error_database["errors"][error_signature] = []
        self.error_database["errors"][error_signature].append(error_info)
        
        # Save to file
        self._save_error_database()
        
        # Log to file
        logging.error(
            f"Error detected:\nType: {error_info['type']}\n"
            f"Message: {error_info['message']}\n"
            f"Context: {error_info['code_context']}"
        )
        
        return error_info

    def find_similar_errors(self, error: Exception, code_context: str) -> Optional[Dict[str, Any]]:
        """Find similar errors in the database that have been successfully resolved."""
        error_signature = self.get_error_signature(error, code_context)
        
        if error_signature in self.error_database["errors"]:
            # Find the most recent successfully resolved error
            for error_instance in reversed(self.error_database["errors"][error_signature]):
                if error_instance["resolved"] and error_instance["successful_solution"]:
                    return error_instance
        
        return None

    def update_error_resolution(self, error_signature: str, solution: str, success: bool):
        """Update the database with the result of an attempted solution."""
        if error_signature in self.error_database["errors"]:
            latest_error = self.error_database["errors"][error_signature][-1]
            latest_error["solutions_attempted"].append({
                "solution": solution,
                "timestamp": datetime.now().isoformat(),
                "success": success
            })
            
            if success:
                latest_error["resolved"] = True
                latest_error["successful_solution"] = solution
            
            self._save_error_database()
            
            logging.info(
                f"Solution {'succeeded' if success else 'failed'} for error: {error_signature}\n"
                f"Solution: {solution}"
            )
