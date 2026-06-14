import json
import requests
from bs4 import BeautifulSoup
import re

class FactChecker:
    def __init__(self, knowledge_base_url):
        self.knowledge_base_url = knowledge_base_url

    def get_facts(self):
        response = requests.get(self.knowledge_base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        facts = []
        for fact_element in soup.find_all('fact'):
            facts.append(fact_element.text.strip())
        return facts

    def check_statement(self, statement):
        facts = self.get_facts()
        contradictions = []
        for fact in facts:
            if re.search(re.escape(statement), fact):
                contradictions.append(fact)
        return contradictions

    def __str__(self):
        return f"FactChecker(knowledge_base_url={self.knowledge_base_url})"

def main():
    knowledge_base_url = "http://example.com/facts"
    fact_checker = FactChecker(knowledge_base_url)
    statement = "The sky is blue"
    contradictions = fact_checker.check_statement(statement)
    print(f"Statement: {statement}")
    print("Contradictions:")
    for contradiction in contradictions:
        print(contradiction)

if __name__ == '__main__':
    main()