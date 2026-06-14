import json
import re
import pathlib
import requests
from bs4 import BeautifulSoup
import psutil
from typing import List, Dict

def validate_task(task: Dict) -> None:
    required_fields = ['name', 'description', 'due_date']
    for field in required_fields:
        if field not in task:
            raise ValueError(f"Missing required field '{field}' in task")

def save_task(task: Dict, tasks_file: pathlib.Path) -> None:
    with open(tasks_file, 'r+') as f:
        tasks = json.load(f)
        tasks.append(task)
        f.seek(0)
        json.dump(tasks, f, indent=2)

def load_tasks(tasks_file: pathlib.Path) -> List[Dict]:
    if not tasks_file.exists():
        tasks_file.write_text('[]')
    with open(tasks_file, 'r') as f:
        return json.load(f)

def get_task_by_name(name: str, tasks: List[Dict]) -> Dict:
    for task in tasks:
        if task['name'] == name:
            return task
    return None

def update_task(task: Dict, tasks_file: pathlib.Path) -> None:
    tasks = load_tasks(tasks_file)
    task_index = next((i for i, t in enumerate(tasks) if t['name'] == task['name']), None)
    if task_index is None:
        raise ValueError(f"Task '{task['name']}' not found")
    tasks[task_index] = task
    save_task(tasks[task_index], tasks_file)

def delete_task(name: str, tasks_file: pathlib.Path) -> None:
    tasks = load_tasks(tasks_file)
    task = get_task