"""
Web scraping module for fetching error solutions from various sources.
"""

import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus
import time
import random

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Configure logging
        logging.basicConfig(
            filename='logs/web_scraper.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def search_stack_overflow(self, error_message: str) -> List[Dict[str, Any]]:
        """
        Search Stack Overflow for solutions to the error.
        
        Args:
            error_message: The error message to search for
            
        Returns:
            List of relevant posts with solutions
        """
        try:
            # Construct search URL
            search_query = quote_plus(f"[python] {error_message}")
            url = f"https://stackoverflow.com/search?q={search_query}"
            
            # Add delay to respect rate limits
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find question links
            for question in soup.select('.question-summary'):
                title = question.select_one('.question-hyperlink')
                votes = question.select_one('.vote-count-post')
                
                if title and votes:
                    question_url = f"https://stackoverflow.com{title['href']}"
                    results.append({
                        'title': title.text,
                        'url': question_url,
                        'votes': int(votes.text),
                        'solutions': self._get_solutions(question_url)
                    })
            
            return sorted(results, key=lambda x: x['votes'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error searching Stack Overflow: {str(e)}")
            return []

    def _get_solutions(self, question_url: str) -> List[Dict[str, Any]]:
        """Extract solutions from a Stack Overflow question page."""
        try:
            # Add delay to respect rate limits
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(question_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            solutions = []
            
            # Find accepted answer first
            accepted_answer = soup.select_one('.answer.accepted-answer')
            if accepted_answer:
                solutions.append(self._parse_answer(accepted_answer, is_accepted=True))
            
            # Find other answers
            for answer in soup.select('.answer:not(.accepted-answer)'):
                solutions.append(self._parse_answer(answer, is_accepted=False))
            
            return sorted(solutions, key=lambda x: x['votes'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error fetching solutions from {question_url}: {str(e)}")
            return []

    def _parse_answer(self, answer_element: BeautifulSoup, is_accepted: bool) -> Dict[str, Any]:
        """Parse a Stack Overflow answer into a structured format."""
        votes = answer_element.select_one('.vote-count-post')
        code_blocks = answer_element.select('code')
        explanation = answer_element.select_one('.post-text')
        
        return {
            'votes': int(votes.text) if votes else 0,
            'is_accepted': is_accepted,
            'code_blocks': [block.text for block in code_blocks],
            'explanation': explanation.text if explanation else '',
        }

    def search_github_issues(self, error_message: str) -> List[Dict[str, Any]]:
        """Search GitHub issues for similar error reports and solutions."""
        try:
            search_query = quote_plus(f"{error_message} language:python")
            url = f"https://github.com/search?q={search_query}&type=issues"
            
            # Add delay to respect rate limits
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for issue in soup.select('.issue-list-item'):
                title = issue.select_one('.issue-title')
                if title:
                    results.append({
                        'title': title.text.strip(),
                        'url': f"https://github.com{title['href']}",
                        'status': 'closed' if 'closed' in issue.get('class', []) else 'open'
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching GitHub issues: {str(e)}")
            return []

    def search_python_docs(self, error_type: str) -> List[Dict[str, Any]]:
        """Search Python documentation for error explanations."""
        try:
            url = f"https://docs.python.org/3/search.html?q={quote_plus(error_type)}"
            
            # Add delay to respect rate limits
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.select('.search-result-item'):
                title = result.select_one('.search-result-title')
                if title:
                    results.append({
                        'title': title.text.strip(),
                        'url': title.find('a')['href'],
                        'description': result.select_one('.search-result-description').text.strip()
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching Python docs: {str(e)}")
            return []
