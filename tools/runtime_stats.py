import time
import json
import psutil
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class RuntimeStats:
    def __init__(self, interval=1, duration=10):
        self.interval = interval
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        self.func_calls = defaultdict(int)

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def record_cpu_usage(self):
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.append(cpu_percent)

    def record_memory_usage(self):
        memory_info = psutil.virtual_memory()
        self.memory_usage.append(memory_info.percent)

    def record_func_call(self, func_name):
        self.func_calls[func_name] += 1

    def save_stats(self, filename):
        stats = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'func_calls': dict(self.func_calls)
        }
        with open(filename, 'w') as f:
            json.dump(stats, f)

def main():
    stats = RuntimeStats(interval=1, duration=10)
    stats.start()

    # Simulated work
    stats.record_func_call('func1')
    time.sleep(1)
    stats.record_func_call('func2')
    time.sleep(1)
    stats.record_func_call('func1')
    time.sleep(1)
    stats.record_func_call