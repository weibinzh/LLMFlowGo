import ast
from typing import List, Dict

def analyze_python_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parse a Python file using the ast module and extract the names and source code of all top-level functions.

    Args:
    file_path: The path of the Python file to analyze.

    Returns:
    A list of objects, each containing 'name' and 'source_code'.
    If the file cannot be parsed, returns an empty list.
    """
    functions_with_source = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in tree.body:
            # Only check top-level nodes that are function definitions
            if isinstance(node, ast.FunctionDef):
                function_info = {
                    "name": node.name,
                    "source_code": ast.unparse(node)
                }
                functions_with_source.append(function_info)

    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
        raise ValueError(f"Failed to analyze Python file due to: {e}")
        
    return functions_with_source
