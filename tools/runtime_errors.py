import os
import json
import re
from pathlib import Path
from typing import List

def extract_last_20_error_logs(log_file: Path) -> List[str]:
    with open(log_file, "r") as file:
        content = file.readlines()
    error_lines = [line for line in content if "ERROR" in line]
    return error_lines[-20:]

def main():
    log_file = Path("logs/helix.log")
    last_20_errors = extract_last_20_error_logs(log_file)
    print(json.dumps(last_20_errors))

if __name__ == "__main__":
    main()