import os
import json
import pathlib
import importlib
import inspect
from typing import Dict, Any

def load_tool(name: str) -> Dict[str, Any]:
    module_path = f"{os.path.dirname(os.path.abspath(__file__))}/tools/{name}.py"
    with open(module_path, "r") as file:
        code = file.read()
    tree = compile(code, filename=module_path, mode='exec')
    loader = importlib.machinery.SourceFileLoader
    namespace = {}
    exec(compile(code, filename=module_path, mode='exec'), namespace)
    return namespace

def test_tool(name: str, namespace: Dict[str, Any]) -> bool:
    test_function_name = "test_function"
    if test_function_name in namespace:
        namespace[test_function_name]()
        return True
    return False

def main():
    tools = os.listdir(f"{os.path.dirname(os.path.abspath(__file__))}/tools")
    results = {}
    for tool in tools:
        if tool.endswith(".py"):
            name = tool[:-3]
            try:
                namespace = load_tool(name)
                is_tested = test_tool(name, namespace)
                results[name] = {"status": "pass" if is_tested else "fail"}
            except Exception as e:
                results[name] = {"status": "fail", "error": str(e)}
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()