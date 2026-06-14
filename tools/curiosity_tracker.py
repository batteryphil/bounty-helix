import json
import jsonlines
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

class CuriosityTracker:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.knowledge = []
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.knowledge = list(jsonlines.Reader(f))
        else:
            print(f"File {self.data_file} not found. Creating a new file.")

    def save_data(self):
        with open(self.data_file, 'a') as f:
            jsonlines.Writer(f).write(self.knowledge)

    def add_knowledge(self, fact: Dict):
        self.knowledge.append(fact)
        self.save_data()

    def get_new_facts(self, date: str) -> List[Dict]:
        new_facts = []
        for fact in self.knowledge:
            if fact.get('timestamp')[:10] == date:
                new_facts.append(fact)
        return new_facts

    def get_new_facts_count(self, date: str) -> int:
        return len(self.get_new_facts(date))

    def get_total_knowledge(self) -> int:
        return len(self.knowledge)

if __name__ == '__main__':
    tracker = CuriosityTracker('curiosity_knowledge.jsonl')
    today = datetime.now().strftime('%Y-%m-%d')
    new_facts = tracker.get_new_facts(today)
    print(f"New facts added on {today}: {new_facts}")