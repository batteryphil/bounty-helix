import json
import jsonlines
import re
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line)
            if query in entry['text']:
                yield entry

def main():
    query = input("Enter a search query: ")
    results = list(search_knowledge(query))
    if results:
        for i, result in enumerate(results[:3], start=1):
            print(f"Result {i}: {result['text']}")
    else:
        print("No matching results found.")

if __name__ == '__main__':
    main()