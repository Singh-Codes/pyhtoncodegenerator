"""
Solution management module for storing, retrieving, and evaluating error solutions.
"""

import sqlite3
from datetime import datetime
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

class SolutionManager:
    def __init__(self, db_path: str = "error_resolution/solutions.db"):
        self.db_path = db_path
        
        # Configure logging
        logging.basicConfig(
            filename='logs/solution_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with necessary tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_signature TEXT UNIQUE,
                    error_type TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_id INTEGER,
                    solution_text TEXT,
                    source TEXT,
                    success_rate REAL,
                    times_attempted INTEGER,
                    times_succeeded INTEGER,
                    last_attempted TIMESTAMP,
                    FOREIGN KEY (error_id) REFERENCES errors (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Database initialization error: {str(e)}")
            raise

    def store_error(self, error_info: Dict[str, Any]) -> int:
        """Store a new error in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if error exists
            cursor.execute(
                "SELECT id FROM errors WHERE error_signature = ?",
                (error_info['signature'],)
            )
            result = cursor.fetchone()
            
            if result:
                error_id = result[0]
                # Update last_seen timestamp
                cursor.execute(
                    "UPDATE errors SET last_seen = ? WHERE id = ?",
                    (datetime.now().isoformat(), error_id)
                )
            else:
                # Insert new error
                cursor.execute('''
                    INSERT INTO errors (error_signature, error_type, first_seen, last_seen)
                    VALUES (?, ?, ?, ?)
                ''', (
                    error_info['signature'],
                    error_info['type'],
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                error_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return error_id
            
        except Exception as e:
            logging.error(f"Error storing error: {str(e)}")
            raise

    def store_solution(self, error_id: int, solution: Dict[str, Any]):
        """Store a new solution for an error."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO solutions (
                    error_id, solution_text, source, success_rate,
                    times_attempted, times_succeeded, last_attempted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_id,
                solution['text'],
                solution['source'],
                0.0,  # Initial success rate
                0,    # Times attempted
                0,    # Times succeeded
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error storing solution: {str(e)}")
            raise

    def get_solutions(self, error_signature: str) -> List[Dict[str, Any]]:
        """Retrieve all solutions for a given error."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.* FROM solutions s
                JOIN errors e ON s.error_id = e.id
                WHERE e.error_signature = ?
                ORDER BY s.success_rate DESC, s.times_succeeded DESC
            ''', (error_signature,))
            
            solutions = []
            for row in cursor.fetchall():
                solutions.append({
                    'id': row[0],
                    'solution_text': row[2],
                    'source': row[3],
                    'success_rate': row[4],
                    'times_attempted': row[5],
                    'times_succeeded': row[6],
                    'last_attempted': row[7]
                })
            
            conn.close()
            return solutions
            
        except Exception as e:
            logging.error(f"Error retrieving solutions: {str(e)}")
            return []

    def update_solution_success(self, solution_id: int, success: bool):
        """Update the success rate of a solution."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current stats
            cursor.execute(
                "SELECT times_attempted, times_succeeded FROM solutions WHERE id = ?",
                (solution_id,)
            )
            result = cursor.fetchone()
            
            if result:
                times_attempted = result[0] + 1
                times_succeeded = result[1] + (1 if success else 0)
                success_rate = times_succeeded / times_attempted
                
                # Update solution stats
                cursor.execute('''
                    UPDATE solutions
                    SET times_attempted = ?,
                        times_succeeded = ?,
                        success_rate = ?,
                        last_attempted = ?
                    WHERE id = ?
                ''', (
                    times_attempted,
                    times_succeeded,
                    success_rate,
                    datetime.now().isoformat(),
                    solution_id
                ))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error updating solution success: {str(e)}")
            raise

    def get_best_solution(self, error_signature: str) -> Optional[Dict[str, Any]]:
        """Get the most successful solution for an error."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.* FROM solutions s
                JOIN errors e ON s.error_id = e.id
                WHERE e.error_signature = ?
                ORDER BY s.success_rate DESC, s.times_succeeded DESC
                LIMIT 1
            ''', (error_signature,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'solution_text': result[2],
                    'source': result[3],
                    'success_rate': result[4],
                    'times_attempted': result[5],
                    'times_succeeded': result[6],
                    'last_attempted': result[7]
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting best solution: {str(e)}")
            return None

    def cleanup_old_solutions(self, days_threshold: int = 30):
        """Remove old, unsuccessful solutions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove solutions that haven't been successful and haven't been tried recently
            cursor.execute('''
                DELETE FROM solutions
                WHERE success_rate < 0.2
                AND julianday('now') - julianday(last_attempted) > ?
                AND times_attempted > 5
            ''', (days_threshold,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error cleaning up old solutions: {str(e)}")
            raise
