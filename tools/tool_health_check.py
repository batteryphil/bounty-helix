import os
import json
import pathlib
import subprocess
import sys

def check_tool_health(tool_name, tool_module):
    try:
        module = __import__(tool_module)
        result = module.check_health()
        return f"{tool_name}: OK - {result}"
    except Exception as e:
        return f"{tool_name}: ERROR - {str(e)}"

def main():
    tools_dir = pathlib.Path(__file__).parent / "tools"
    health_report = []

    for tool_name in os.listdir(tools_dir):
        if tool_name.endswith(".py") and not tool_name.startswith("_"):
            tool_path = tools_dir / tool_name
            tool_module = f"tools.{tool_name[:-3]}"
            health_report.append(check_tool_health(tool_name, tool_module))

    print(json.dumps({"health_report": health_report}, indent=2))

if __name__ == "__main__":
    main()