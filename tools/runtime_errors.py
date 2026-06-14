import os
import json
import re
from pathlib import Path
from typing import List

def get_last_20_error_lines(log_file: Path) -> List[str]:
    with open(log_file, 'r') as file:
        lines = file.readlines()
    error_lines = [line for line in lines if 'ERROR' in line]
    return error_lines[-20:] if len(error_lines) > 20 else error_lines

def main():
    log_file = Path('/path/to/helix/log/file.log')
    error_lines = get_last_20_error_lines(log_file)
    print(json.dumps(error_lines))

if __name__ == '__main__':
    main()