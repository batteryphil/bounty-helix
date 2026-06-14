import json
import re
import pathlib
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

class NoveltyDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.novel_items = defaultdict(int)
        self.load_data()

    def load_data(self):
        try:
            with open(self.data_path, 'r') as f:
                self.old_data = json.load(f)
        except FileNotFoundError:
            self.old_data = {}

    def save_data(self):
        with open(self.data_path, 'w') as f:
            json.dump(self.novel_items, f)

    def analyze_response(self, response):
        words = re.findall(r'\b\w+\b', response.lower())
        for word in words:
            if word not in self.old_data:
                self.novel_items[word] += 1