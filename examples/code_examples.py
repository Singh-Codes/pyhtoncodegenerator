"""
Example code generation prompts and test cases.
"""

CODING_EXAMPLES = [
    {
        "category": "Basic Functions",
        "examples": [
            {
                "prompt": "Write a Python function to find the factorial of a number",
                "reference": """
def factorial(n: int) -> int:
    \"\"\"Calculate the factorial of a number.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
            },
            {
                "prompt": "Create a function to check if a string is palindrome",
                "reference": """
def is_palindrome(s: str) -> bool:
    \"\"\"Check if a string is palindrome.\"\"\"
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
"""
            }
        ]
    },
    {
        "category": "Data Structures",
        "examples": [
            {
                "prompt": "Implement a stack data structure with push, pop, and peek operations",
                "reference": """
class Stack:
    \"\"\"Implementation of a stack data structure.\"\"\"
    def __init__(self):
        self.items = []
    
    def push(self, item: any) -> None:
        \"\"\"Add item to stack.\"\"\"
        self.items.append(item)
    
    def pop(self) -> any:
        \"\"\"Remove and return top item.\"\"\"
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek(self) -> any:
        \"\"\"Return top item without removing.\"\"\"
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek at empty stack")
    
    def is_empty(self) -> bool:
        \"\"\"Check if stack is empty.\"\"\"
        return len(self.items) == 0
"""
            },
            {
                "prompt": "Create a binary search tree class with insert and search methods",
                "reference": """
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    \"\"\"Implementation of a binary search tree.\"\"\"
    def __init__(self):
        self.root = None
    
    def insert(self, value: int) -> None:
        \"\"\"Insert a value into the BST.\"\"\"
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node: Node, value: int) -> None:
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value: int) -> bool:
        \"\"\"Search for a value in the BST.\"\"\"
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node: Node, value: int) -> bool:
        if node is None:
            return False
        if node.value == value:
            return True
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)
"""
            }
        ]
    },
    {
        "category": "Algorithms",
        "examples": [
            {
                "prompt": "Implement merge sort algorithm",
                "reference": """
def merge_sort(arr: list) -> list:
    \"\"\"Sort array using merge sort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: list, right: list) -> list:
    \"\"\"Merge two sorted arrays.\"\"\"
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
            },
            {
                "prompt": "Write a function to find the longest common subsequence of two strings",
                "reference": """
def longest_common_subsequence(text1: str, text2: str) -> str:
    \"\"\"Find the longest common subsequence of two strings.\"\"\"
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
"""
            }
        ]
    },
    {
        "category": "API and Web",
        "examples": [
            {
                "prompt": "Create a FastAPI endpoint for user registration",
                "reference": """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import hashlib

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

@app.post("/register")
async def register_user(user: User):
    \"\"\"Register a new user.\"\"\"
    try:
        # Hash password
        hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
        
        # In real app, save to database
        new_user = {
            "username": user.username,
            "email": user.email,
            "password": hashed_password,
            "full_name": user.full_name
        }
        
        return {"message": "User registered successfully", "user": new_user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""
            },
            {
                "prompt": "Write a function to fetch and parse JSON from a REST API",
                "reference": """
import aiohttp
import asyncio
from typing import Dict, Any

async def fetch_api_data(url: str) -> Dict[str, Any]:
    \"\"\"Fetch and parse JSON data from API.\"\"\"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API request failed with status {response.status}")
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return {}

# Example usage
async def main():
    api_url = "https://api.example.com/data"
    data = await fetch_api_data(api_url)
    print(data)

if __name__ == "__main__":
    asyncio.run(main())
"""
            }
        ]
    }
]

def get_example_by_category(category: str):
    """Get examples for a specific category."""
    for cat in CODING_EXAMPLES:
        if cat["category"].lower() == category.lower():
            return cat["examples"]
    return []

def get_all_prompts():
    """Get all available prompts."""
    prompts = []
    for category in CODING_EXAMPLES:
        for example in category["examples"]:
            prompts.append({
                "category": category["category"],
                "prompt": example["prompt"]
            })
    return prompts

def get_reference_solution(prompt: str):
    """Get reference solution for a prompt."""
    for category in CODING_EXAMPLES:
        for example in category["examples"]:
            if example["prompt"] == prompt:
                return example["reference"]
    return None
