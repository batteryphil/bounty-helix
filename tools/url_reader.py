import requests
import json
from bs4 import BeautifulSoup
import re
import pathlib
from pathlib import Path
import psutil

class URLReader:
    def __init__(self, url):
        self.url = url
        self.content = None
        self.soup = None
        self.title = None
        self.text = None

    def fetch_content(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            self.content = response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {self.url}: {e}")

    def parse_html(self):
        self.soup = BeautifulSoup(self.content, 'html.parser')
        self.title = self.soup.title.string
        self.text = self.soup.get_text()

    def save_data(self, output_file):
        data = {
            "url": self.url,
            "title": self.title,
            "text": self.text
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)