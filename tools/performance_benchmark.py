import json
import os
import sys
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup
from psutil import cpu_percent

class PerformanceBenchmark:
    def __init__(self, reference_responses: List[str], generated_responses: List[str]):
        self.reference_responses = reference_responses
        self.generated_responses = generated_responses
        self.scores = []

    def score_response(self, ref: str, gen: str) -> float:
        # Implement your scoring logic here
        # Return a score between 0 and 1
        pass

    def benchmark(self) -> None:
        for ref, gen in zip(self.reference_responses, self.generated_responses):
            score = self.score_response(ref, gen)
            self.scores.append(score)