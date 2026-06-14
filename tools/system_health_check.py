import json
import psutil
import os
import sys
from pathlib import Path
from typing import Dict

def get_system_metrics() -> Dict[str, float]:
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    network_io_counters = psutil.net_io_counters()._asdict()
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'disk_percent': disk_percent,
        'network_sent_bytes': network_io_counters.get('bytes_sent', 0),
        'network_recv_bytes': network_io_counters.get('bytes_recv', 0),
    }

def check_thresholds(metrics: Dict[str, float]) -> str:
    thresholds = {
        'cpu_percent': 75,
        'memory_percent': 80,
        'disk_percent': 90,
    }
    status = 'ok'
    for key, threshold in thresholds.items():
        if metrics[key] > threshold:
            status = 'warning' if status == 'ok' else 'critical'
            print(f"Threshold exceeded for {key}: {metrics[key]}")
    return status

def main():
    metrics = get_system_metrics()
    status = check_thresholds(metrics)
    print(json.dumps(metrics))
    print(status)

if __name__ == '__main__':
    main()