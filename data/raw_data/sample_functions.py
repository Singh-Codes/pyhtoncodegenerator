"""Sample Python functions for training the code generator."""

def factorial(n: int) -> int:
    """
    Calculate the factorial of a number.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome.
    
    Args:
        s (str): The string to check
        
    Returns:
        bool: True if the string is a palindrome, False otherwise
    """
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

class BinarySearchTree:
    """
    A binary search tree implementation.
    """
    
    def __init__(self):
        """Initialize an empty binary search tree."""
        self.root = None
        
    def insert(self, value: int) -> None:
        """
        Insert a value into the binary search tree.
        
        Args:
            value (int): The value to insert
        """
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node: 'Node', value: int) -> None:
        """
        Recursively insert a value into the binary search tree.
        
        Args:
            node (Node): The current node
            value (int): The value to insert
        """
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

class Node:
    """A node in the binary search tree."""
    
    def __init__(self, value: int):
        """
        Initialize a new node.
        
        Args:
            value (int): The value to store in the node
        """
        self.value = value
        self.left = None
        self.right = None
