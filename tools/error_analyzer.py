import json
import re
import pathlib
from typing import List, Dict

def parse_error_log(log_file: pathlib.Path) -> List[str]:
    with open(log_file, 'r') as f:
        log_content = f.read()
    error_lines = re.findall(r'ERROR: (.+)', log_content)
    return error_lines[-20:]

def count_errors(errors: List[str]) -> Dict[str, int]:
    error_count = {}
    for error in errors:
        error = error.strip()
        error_count[error] = error_count.get(error, 0) + 1
    return error_count

def most_common_errors(error_count: Dict[str, int]) -> List[tuple]:
    return sorted(error_count.items(), key=lambda x: x[1], reverse=True)[:5]

def main():
    log_file = pathlib.Path('error.log')
    errors = parse_error_log(log_file)
    error_count = count_errors(errors)
    most_common = most_common_errors(error_count)
    print(json.dumps(most_common, indent=2))

if __name__ == '__main__':
    main()