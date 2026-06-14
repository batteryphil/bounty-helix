import requests
from bs4 import BeautifulSoup
import json
import re
import pathlib
import psutil

class WebResearch:
    def __init__(self, query):
        self.query = query
        self.results = []

    def search(self):
        url = f"https://www.google.com/search?q={self.query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        search_results = soup.find_all("div", {"class": "yuRUdf"})
        for result in search_results:
            link = result.find("a")["href"]
            self.results.append({"title": result.find("h3").text, "link": link})

    def extract_info(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("title").text
        content = soup.find("div", {"id": "main-content"})
        if content:
            text = content.get_text()
            paragraphs = re.findall(r'<p>(.*?)</p>', text, re.DOTALL)
            paragraphs = [p.replace('<br/>', ' ').replace('<br>', ' ') for p in paragraphs]
            paragraphs = [p.replace('<span>', ' ').replace('</span>', ' ') for p in paragraphs]
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            text = '\n\n'.join(paragraphs)
            text = re.sub(r'\n+', '\n', text)
        else:
            text = ""
        return {"title": title, "content": text}