"""
Collects and stores performance metrics across sessions.

Metrics collected:
- response_time: Time taken to generate a response in seconds
- accuracy: Percentage of correct responses out of total responses
- fitness_score: Fitness score assigned by the human for the generated response

Metrics are stored in a JSON file named 'metrics.json' in the current directory.
"""

import json
import time
import os

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "accuracies": [],
            "fitness_scores": []
        }

    def log_metrics(self, response_time, accuracy, fitness_score):
        self.metrics["response_times"].append(response_time)
        self.metrics["accuracies"].append(accuracy)
        self.metrics["fitness_scores"].append(fitness_score)

    def save_metrics(self):
        with open("metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)

def register_tool():
    from helix.registry import ToolRegistry
    ToolRegistry.register_tool("metrics_collector", MetricsCollector)

register_tool()