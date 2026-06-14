import json
import psutil
import os
import time
import re
import pathlib
from pathlib import Path
import requests
from bs4 import BeautifulSoup

class SystemHealth:
    def __init__(self, cpu_threshold, ram_threshold, disk_threshold):
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.disk_threshold = disk_threshold
        self.last_cpu_usage = None
        self.last_ram_usage = None
        self.last_disk_usage = None

    def check_cpu_usage(self):
        cpu_usage = psutil.cpu_percent()
        if self.last_cpu_usage is not None and cpu_usage > self.cpu_threshold:
            print(f"High CPU usage detected: {cpu_usage}%")
        self.last_cpu_usage = cpu_usage

    def check_ram_usage(self):
        ram_usage = psutil.virtual_memory()
        self.last_ram_usage = ram_usage.percent

    def check_disk_usage(self):
        disk_usage = psutil.disk_usage('/')
        self.last_disk_usage = disk_usage.percent

    def get_system_health(self):
        system_health = {
            'cpu_usage': self.last_cpu_usage,
            'ram_usage': self.last_ram_usage,
            'disk_usage': self.last_disk_usage
        }
        return json.dumps(system_health)