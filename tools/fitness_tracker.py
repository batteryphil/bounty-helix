import json
import os
import pathlib
import datetime
import re
from typing import List, Dict

def save_fitness(fitness: float, filename: str):
    with open(filename, 'w') as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(), "fitness": fitness}, f)

def load_fitness(filename: str) -> List[Dict]:
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_fitness(filename: str):
    fitness_data = load_fitness(filename)
    if not fitness_data:
        return "No fitness data available"
    
    fitness_values = [entry['fitness'] for entry in fitness_data]
    last_entry = fitness_data[-1]
    current_fitness = last_entry['fitness']
    
    if len(fitness_values) < 7:
        return f"Current fitness data is too short to analyze. Only {len(fitness_values)} entries available."
    else:
        avg_fitness = sum(fitness_values[-7:]) / 7
        return f"Current fitness: {current_fitness}\nAverage fitness (last 7 days): {avg_fitness}"