import json
import requests
from bs4 import BeautifulSoup
import re
import jsonlines
import pathlib
import psutil
import time
from typing import List, Dict

def fetch_articles(urls: List[str]) -> List[str]:
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article = '\n'.join([p.text for p in paragraphs])
        articles.append(article)
    return articles

def filter_articles(articles: List[str], heuristics: List[str]) -> List[str]:
    filtered_articles = []
    for article in articles:
        for heuristic in heuristics:
            if re.search(heuristic, article):
                filtered_articles.append(article)
                break
    return filtered_articles

def save_to_file(articles: List[str], file_path: str):
    with jsonlines.open(file_path, mode='a') as f:
        for article in articles:
            f.write(article)

def main():
    sources = [
        'https://example.com/source1',
        'https://example.com/source2',
        # Add more trusted sources here
    ]
    heuristics = [
        r'example_heuristic1',
        r'example_heuristic2',
        # Add more heuristics here
    ]
    
    articles = fetch_articles(sources)
    filtered_articles = filter_articles(articles, heuristics)
    save_to_file(filtered_articles, 'curiosity_knowledge.jsonl')

if __name__ == '__main__':
    main()