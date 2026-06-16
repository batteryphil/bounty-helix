import json
import jsonlines
import re
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        results = []
        for line in f:
            entry = json.loads(line)
            if any(query.lower() in str(entry['text']).lower() for query in query.split()):
                results.append(entry)
            if len(results) >= 3:
                break
        return results

def main():
    query = input("Enter a search query: ")
    results = search_knowledge(query)
    print("Top 3 relevant results:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['text']}")

if __name__ == '__main__':
    main()