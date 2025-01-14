"""
Main error resolution module that coordinates error detection, solution finding, and application.
"""

import logging
from typing import Dict, Any, Optional, List
import traceback
from pathlib import Path

from .error_detector import ErrorDetector
from .web_scraper import WebScraper
from .nlp_analyzer import NLPAnalyzer
from .solution_manager import SolutionManager

class ErrorResolver:
    def __init__(self):
        # Initialize components
        self.error_detector = ErrorDetector()
        self.web_scraper = WebScraper()
        self.nlp_analyzer = NLPAnalyzer()
        self.solution_manager = SolutionManager()
        
        # Configure logging
        logging.basicConfig(
            filename='logs/error_resolver.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def handle_error(self, error: Exception, code_context: str) -> Optional[Dict[str, Any]]:
        """
        Main error handling method that coordinates the resolution process.
        
        Args:
            error: The exception that occurred
            code_context: The code that caused the error
            
        Returns:
            Dict containing the solution if found, None otherwise
        """
        try:
            # Step 1: Log the error
            error_info = self.error_detector.log_error(error, code_context)
            
            # Step 2: Check for similar errors in database
            similar_error = self.error_detector.find_similar_errors(error, code_context)
            if similar_error and similar_error["successful_solution"]:
                logging.info(f"Found previous solution for error: {error}")
                return similar_error
            
            # Step 3: Extract error context using NLP
            error_context = self.nlp_analyzer.extract_error_context(
                str(error),
                traceback.format_exc()
            )
            
            # Step 4: Search for solutions
            solutions = self._search_solutions(error_context)
            
            # Step 5: Rank and analyze solutions
            ranked_solutions = self.nlp_analyzer.rank_solutions(error_context, solutions)
            
            if ranked_solutions:
                best_solution = ranked_solutions[0]
                return self._process_solution(best_solution, error_info)
            
            return None
            
        except Exception as e:
            logging.error(f"Error in handle_error: {str(e)}")
            return None

    def _search_solutions(self, error_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for solutions from various sources."""
        solutions = []
        
        # Search Stack Overflow
        so_solutions = self.web_scraper.search_stack_overflow(
            f"{error_context['error_type']}: {error_context['processed_message']}"
        )
        solutions.extend(so_solutions)
        
        # Search GitHub issues
        github_solutions = self.web_scraper.search_github_issues(
            f"{error_context['error_type']}: {error_context['processed_message']}"
        )
        solutions.extend(github_solutions)
        
        # Search Python docs
        doc_solutions = self.web_scraper.search_python_docs(error_context['error_type'])
        solutions.extend(doc_solutions)
        
        return solutions

    def _process_solution(self, solution: Dict[str, Any], error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process and store a solution."""
        # Evaluate solution quality
        quality_score = self.nlp_analyzer.evaluate_solution_quality(solution)
        
        # Store solution if quality is good enough
        if quality_score > 0.5:
            error_id = self.solution_manager.store_error(error_info)
            self.solution_manager.store_solution(error_id, {
                'text': solution.get('explanation', ''),
                'source': 'web_search',
                'quality_score': quality_score
            })
        
        return {
            'solution': solution.get('explanation', ''),
            'code_blocks': solution.get('code_blocks', []),
            'source_url': solution.get('url', ''),
            'confidence_score': quality_score
        }

    def apply_solution(self, solution: Dict[str, Any], error_context: str) -> bool:
        """
        Attempt to apply a solution to the error.
        
        Args:
            solution: The solution to apply
            error_context: The context in which the error occurred
            
        Returns:
            bool indicating whether the solution was successfully applied
        """
        try:
            # Extract code blocks from solution
            code_blocks = solution.get('code_blocks', [])
            if not code_blocks:
                return False
            
            # Try to apply each code block
            for code_block in code_blocks:
                try:
                    # Create a safe execution environment
                    local_vars = {}
                    global_vars = {'__builtins__': __builtins__}
                    
                    # Execute the solution code
                    exec(code_block, global_vars, local_vars)
                    
                    # If we get here, the code executed successfully
                    return True
                    
                except Exception as e:
                    logging.warning(f"Failed to apply solution code block: {str(e)}")
                    continue
            
            return False
            
        except Exception as e:
            logging.error(f"Error applying solution: {str(e)}")
            return False

    def learn_from_attempt(self, error_signature: str, solution_id: int, success: bool):
        """Learn from the success or failure of a solution attempt."""
        try:
            # Update solution success rate
            self.solution_manager.update_solution_success(solution_id, success)
            
            # Clean up old unsuccessful solutions periodically
            self.solution_manager.cleanup_old_solutions()
            
        except Exception as e:
            logging.error(f"Error learning from attempt: {str(e)}")

    def get_solution_stats(self) -> Dict[str, Any]:
        """Get statistics about solution effectiveness."""
        try:
            conn = self.solution_manager._get_connection()
            cursor = conn.cursor()
            
            # Get overall stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_solutions,
                    AVG(success_rate) as avg_success_rate,
                    SUM(times_succeeded) as total_successes,
                    SUM(times_attempted) as total_attempts
                FROM solutions
            ''')
            
            stats = cursor.fetchone()
            conn.close()
            
            return {
                'total_solutions': stats[0],
                'average_success_rate': stats[1],
                'total_successes': stats[2],
                'total_attempts': stats[3]
            }
            
        except Exception as e:
            logging.error(f"Error getting solution stats: {str(e)}")
            return {
                'total_solutions': 0,
                'average_success_rate': 0,
                'total_successes': 0,
                'total_attempts': 0
            }
