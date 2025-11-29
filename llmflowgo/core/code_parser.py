import ast
from typing import List, Dict, Optional, Any

class FunctionVisitor(ast.NodeVisitor):
    """
    An AST visitor that finds other functions called within a specific function.
    """
    def __init__(self, target_function_name: str):
        self.target_function_name = target_function_name
        self.called_functions = set()
        self.in_target_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self.target_function_name:
            self.in_target_function = True
            self.generic_visit(node)
            self.in_target_function = False
        else:
            pass

    def visit_Call(self, node: ast.Call):
        if self.in_target_function:
            if isinstance(node.func, ast.Name):
                self.called_functions.add(node.func.id)
        self.generic_visit(node)


def get_function_calls(tree: ast.AST, target_function_name: str) -> List[str]:
    """
    Find all function names called within a specific function from the AST.
    """
    visitor = FunctionVisitor(target_function_name)
    visitor.visit(tree)
    return list(visitor.called_functions)


def get_functions_details(tree: ast.AST, function_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extract detailed information for specified functions from the AST.
    """
    details = []
    
    # Create a set of function names for quick lookup
    name_set = set(function_names)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in name_set:
            # Extract parameters
            params = []
            for arg in node.args.args:
                param_str = arg.arg
                if arg.annotation:
                    param_str += f": {ast.unparse(arg.annotation)}"
                params.append(param_str)

            # Extract return type
            returns = ast.unparse(node.returns) if node.returns else "Not specified"
            
            # Extract docstring as a temporary summary
            summary = ast.get_docstring(node) or "No summary available."

            details.append({
                "name": node.name,
                "parameters": f"({', '.join(params)})",
                "returns": returns,
                "summary": summary.strip(),
                "source_code": ast.unparse(node)
            })
    
    return details
