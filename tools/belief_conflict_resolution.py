import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class Belief:
    def __init__(self, identifier: str, confidence: float, timestamp: datetime):
        self.identifier = identifier
        self.confidence = confidence
        self.timestamp = timestamp

def load_beliefs(file_path: Path) -> List[Belief]:
    with open(file_path, 'r') as file:
        beliefs = json.load(file)
    return [Belief(bel['identifier'], bel['confidence'], datetime.fromisoformat(bel['timestamp'])) for bel in beliefs]

def resolve_conflicts(beliefs: List[Belief]) -> List[Belief]:
    resolved_beliefs = []
    for belief in beliefs:
        conflicting_beliefs = [b for b in beliefs if b.identifier == belief.identifier and b != belief]
        if conflicting_beliefs:
            resolved = max(conflicting_beliefs, key=lambda b: (-b.confidence, b.timestamp))
            resolved_beliefs.append(resolved)
        else:
            resolved_beliefs.append(belief)
    return resolved_beliefs

def main():
    beliefs_file = Path('beliefs.json')
    beliefs = load_beliefs(beliefs_file)
    resolved = resolve_conflicts(beliefs)
    print(json.dumps([bel.__dict__ for bel in resolved], indent=2))

if __name__ == '__main__':
    main()