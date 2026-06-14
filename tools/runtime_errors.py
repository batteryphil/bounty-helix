import json
import re
import pathlib
from typing import List, Dict

def parse_log_file(log_file_path: str) -> List[Dict]:
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    error_messages = []
    for line in lines[-20:]:  # Get the last 20 lines
        if 'ERROR' in line:
            error_messages.append(line.strip())
    return error_messages

def main():
    log_file_path = 'path/to/helix/log/file.log'
    error_messages = parse_log_file(log_file_path)
    print(json.dumps(error_messages, indent=2))

if __name__ == '__main__':
    main()