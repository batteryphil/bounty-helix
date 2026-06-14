import json
import os
import sys
import time
import json
import re
import pathlib
import psutil
import requests
from bs4 import BeautifulSoup

class BehaviorTracker:
    def __init__(self):
        self.response_success_rate = []
        self.hallucination_rate = []
        self.novel_belief_generation = []
        self.timestamp = []

    def log_response_success_rate(self, success):
        self.response_success_rate.append(success)

    def log_hallucination_rate(self, hallucinations):
        self.hallucination_rate.append(hallucinations)

    def log_novel_belief(self, novel_belief):
        self.novel_belief_generation.append(novel_belief)

    def log_timestamp(self):
        self.timestamp.append(time.time())

    def save_data(self, filename):
        data = {
            'response_success_rate': self.response_success_rate,
            'hallucination_rate': self.hallucination_rate,
            'novel_belief_generation': self.novel_belief_generation,
            'timestamp': self.timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)