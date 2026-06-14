import json
import os
from pathlib import Path
from typing import List, Tuple

def load_beliefs(filename: str) -> List[Tuple[str, float]]:
    with open(filename, 'r') as f:
        beliefs = json.load(f)
    return [(belief['statement'], belief['confidence']) for belief in beliefs if belief['confidence'] > 0.8]

def find_conflicts(beliefs: List[Tuple[str, float]]) -> List[Tuple[Tuple[str, float], Tuple[str, float]]]:
    conflicts = []
    for i, (belief1, _) in enumerate(beliefs):
        for j, (belief2, _) in enumerate(beliefs[i+1:]):
            if belief1[0] != belief2[0] and belief1[0] != 'None' and belief2[0] != 'None':
                if 'or' in belief1[0].lower() or 'or' in belief2[0].lower() or 'and' in belief1[0].lower() or 'and' in belief2[0].lower():
                    conflicts.append((belief1, belief2))
    return conflicts