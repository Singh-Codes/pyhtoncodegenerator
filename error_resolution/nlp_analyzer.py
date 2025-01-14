"""
NLP module for analyzing error messages and solutions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from collections import Counter

class NLPAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Configure logging
        logging.basicConfig(
            filename='logs/nlp_analyzer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove code snippets
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text

    def extract_error_context(self, error_message: str, traceback: str) -> Dict[str, Any]:
        """Extract relevant information from error message and traceback."""
        try:
            # Extract error type
            error_type_match = re.search(r'^([A-Za-z.]+Error):', error_message)
            error_type = error_type_match.group(1) if error_type_match else "Unknown"
            
            # Extract line number
            line_num_match = re.search(r'line (\d+)', traceback)
            line_number = int(line_num_match.group(1)) if line_num_match else None
            
            # Extract relevant code
            code_match = re.search(r'File ".*", line \d+.*\n\s*(.*)\n', traceback)
            relevant_code = code_match.group(1).strip() if code_match else None
            
            return {
                "error_type": error_type,
                "line_number": line_number,
                "relevant_code": relevant_code,
                "processed_message": self.preprocess_text(error_message)
            }
            
        except Exception as e:
            logging.error(f"Error extracting context: {str(e)}")
            return {
                "error_type": "Unknown",
                "line_number": None,
                "relevant_code": None,
                "processed_message": self.preprocess_text(error_message)
            }

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        try:
            # Preprocess texts
            processed_text1 = self.preprocess_text(text1)
            processed_text2 = self.preprocess_text(text2)
            
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def rank_solutions(self, error_context: Dict[str, Any], solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank potential solutions based on relevance to the error."""
        try:
            ranked_solutions = []
            error_text = f"{error_context['error_type']} {error_context['processed_message']}"
            
            for solution in solutions:
                # Combine solution text
                solution_text = f"{solution.get('title', '')} {solution.get('explanation', '')}"
                
                # Calculate similarity score
                similarity = self.calculate_similarity(error_text, solution_text)
                
                # Add score to solution dict
                solution_with_score = solution.copy()
                solution_with_score['relevance_score'] = similarity
                ranked_solutions.append(solution_with_score)
            
            # Sort by relevance score
            ranked_solutions.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return ranked_solutions
            
        except Exception as e:
            logging.error(f"Error ranking solutions: {str(e)}")
            return solutions

    def extract_key_concepts(self, text: str, n: int = 5) -> List[str]:
        """Extract key concepts/terms from text."""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize and count terms
            words = processed_text.split()
            word_freq = Counter(words)
            
            # Remove common programming terms
            common_terms = {'python', 'error', 'file', 'line', 'code', 'function', 'class'}
            for term in common_terms:
                word_freq.pop(term, None)
            
            # Get most common terms
            return [word for word, _ in word_freq.most_common(n)]
            
        except Exception as e:
            logging.error(f"Error extracting key concepts: {str(e)}")
            return []

    def evaluate_solution_quality(self, solution: Dict[str, Any]) -> float:
        """Evaluate the quality of a solution based on various factors."""
        try:
            score = 0.0
            
            # Factor 1: Presence of code examples
            if solution.get('code_blocks'):
                score += 0.3
            
            # Factor 2: Length of explanation (not too short, not too long)
            explanation_length = len(solution.get('explanation', '').split())
            if 50 <= explanation_length <= 500:
                score += 0.2
            
            # Factor 3: Votes/acceptance
            if solution.get('is_accepted'):
                score += 0.3
            
            votes = solution.get('votes', 0)
            if votes > 0:
                score += min(0.2, votes / 100)  # Cap at 0.2
            
            return score
            
        except Exception as e:
            logging.error(f"Error evaluating solution quality: {str(e)}")
            return 0.0
