"""
Error Resolution System for Python Code Generator AI.

This package provides autonomous error resolution capabilities by:
1. Detecting and logging errors
2. Searching the web for solutions
3. Using NLP to understand errors and solutions
4. Managing a database of past solutions
5. Learning from successful and failed attempts
"""

from .error_detector import ErrorDetector
from .web_scraper import WebScraper
from .nlp_analyzer import NLPAnalyzer
from .solution_manager import SolutionManager
from .error_resolver import ErrorResolver

__all__ = [
    'ErrorDetector',
    'WebScraper',
    'NLPAnalyzer',
    'SolutionManager',
    'ErrorResolver'
]
