import json
import os
from pathlib import Path
from typing import List, Tuple

# Load beliefs from file
def load_beliefs(file_path: Path) -> List[dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

# Extract belief confidence from belief dict
def get_confidence(belief: dict) -> float:
    return belief['confidence']

# Check if two beliefs contradict each other
def contradict(belief1: dict, belief2: dict) -> bool:
    if belief1['subject'] == belief2['subject']:
        return belief1['predicate'] != belief2['predicate']

# Find pairs of beliefs with high confidence that contradict each other
def find_conflict_pairs(beliefs: List[dict], confidence_threshold: float) -> List[Tuple[dict, dict]]:
    return [(b1, b2) for b1, b2 in combinations(beliefs, 2)
           if get_confidence(b1) > confidence_threshold and get_confidence(b2) > confidence_threshold and contradict(b1, b2)]