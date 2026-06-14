import os
import json
import re
import pathlib
from pathlib import Path
from typing import List, Tuple

def search_files(directory: str, pattern: str) -> List[Tuple[str, int]]:
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.py'):
                with open(file_path, 'r') as f:
                    content = f.read()
                    if re.search(pattern, content):
                        matches.append((file_path, content.index(pattern)))
    return matches

def main():
    directory = 'path/to/project'
    pattern = 'search_pattern'
    results = search_files(directory, pattern)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()