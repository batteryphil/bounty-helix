import sys
import logging
import json
from pathlib import Path
from requests import RequestException
from bs4 import BeautifulSoup
from json import JSONDecodeError
from re import Pattern, match
from psutil import Process

class ErrorHandler:
    def __init__(self):
        self.error_logs = []
        self.error_counts = {}
        self.process = Process.current()

    def register_tool(self, tool_name):
        self.error_counts[tool_name] = 0

    def log_error(self, tool_name, exc):
        self.error_counts[tool_name] += 1
        error_log = {
            'timestamp': self.process.cpu_times(),
            'tool': tool_name,
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'traceback': sys.exc_info()[2]
        }
        self.error_logs.append(error_log)