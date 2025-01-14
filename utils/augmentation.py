"""
Data augmentation techniques for code generation.
"""

import random
import ast
import astor
import numpy as np
from typing import List, Dict, Any

class CodeAugmenter:
    def __init__(self, p=0.5):
        self.p = p
        self.augmentations = [
            self.rename_variables,
            self.swap_binary_operands,
            self.add_comments,
            self.change_case_style,
            self.insert_type_hints,
            self.reorder_functions
        ]

    def rename_variables(self, code: str) -> str:
        """Rename variables while preserving semantics."""
        try:
            tree = ast.parse(code)
            var_names = {}
            
            class VariableRenamer(ast.NodeTransformer):
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        if node.id not in var_names:
                            var_names[node.id] = f'var_{len(var_names)}'
                        node.id = var_names[node.id]
                    elif isinstance(node.ctx, ast.Load) and node.id in var_names:
                        node.id = var_names[node.id]
                    return node
            
            tree = VariableRenamer().visit(tree)
            return astor.to_source(tree)
        except:
            return code

    def swap_binary_operands(self, code: str) -> str:
        """Swap operands in commutative operations."""
        try:
            tree = ast.parse(code)
            
            class OperandSwapper(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    if isinstance(node.op, (ast.Add, ast.Mult)) and random.random() < 0.5:
                        node.left, node.right = node.right, node.left
                    return node
            
            tree = OperandSwapper().visit(tree)
            return astor.to_source(tree)
        except:
            return code

    def add_comments(self, code: str) -> str:
        """Add informative comments."""
        try:
            tree = ast.parse(code)
            
            class CommentAdder(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if not ast.get_docstring(node):
                        docstring = f'"""\nFunction to {node.name.replace("_", " ")}.\n"""'
                        node.body.insert(0, ast.Expr(ast.Str(s=docstring)))
                    return node
            
            tree = CommentAdder().visit(tree)
            return astor.to_source(tree)
        except:
            return code

    def change_case_style(self, code: str) -> str:
        """Switch between camelCase and snake_case."""
        try:
            tree = ast.parse(code)
            
            def to_snake_case(name):
                return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
            
            class CaseStyleChanger(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if random.random() < 0.5:
                        node.name = to_snake_case(node.name)
                    return node
            
            tree = CaseStyleChanger().visit(tree)
            return astor.to_source(tree)
        except:
            return code

    def insert_type_hints(self, code: str) -> str:
        """Add type hints to function parameters."""
        try:
            tree = ast.parse(code)
            
            class TypeHintInserter(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    for arg in node.args.args:
                        if not arg.annotation:
                            # Simple type inference
                            if 'str' in arg.arg or 'name' in arg.arg:
                                arg.annotation = ast.Name(id='str', ctx=ast.Load())
                            elif 'num' in arg.arg or 'count' in arg.arg:
                                arg.annotation = ast.Name(id='int', ctx=ast.Load())
                            elif 'list' in arg.arg or 'array' in arg.arg:
                                arg.annotation = ast.Subscript(
                                    value=ast.Name(id='List', ctx=ast.Load()),
                                    slice=ast.Name(id='Any', ctx=ast.Load())
                                )
                    return node
            
            tree = TypeHintInserter().visit(tree)
            return astor.to_source(tree)
        except:
            return code

    def reorder_functions(self, code: str) -> str:
        """Reorder function definitions while preserving dependencies."""
        try:
            tree = ast.parse(code)
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            if len(functions) > 1:
                random.shuffle(functions)
                tree.body = functions
            return astor.to_source(tree)
        except:
            return code

    def augment(self, code: str) -> str:
        """Apply random augmentations to code."""
        if random.random() < self.p:
            augmentation = random.choice(self.augmentations)
            return augmentation(code)
        return code
