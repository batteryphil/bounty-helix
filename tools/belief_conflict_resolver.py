import json
from typing import List, Dict

def resolve_conflict(beliefs: List[Dict]) -> Dict:
    # Find belief with highest confidence
    max_confidence = 0
    consensus_belief = None
    for belief in beliefs:
        if belief['confidence'] > max_confidence and not any(other_belief['belief'] == belief['belief'] and other_belief['confidence'] > belief['confidence'] for other_belief in beliefs if other_belief != belief):
            max_confidence = belief['confidence']
            consensus_belief = belief
    
    return consensus_belief

def main():
    beliefs = [
        {'belief': 'The sky is blue', 'confidence': 0.8},
        {'belief': 'The sky is green', 'confidence': 0.3},
        {'belief': 'The sky is blue', 'confidence': 0.6},
        {'belief': 'The sky is sometimes gray', 'confidence': 0.4}
    ]
    
    consensus = resolve_conflict(beliefs)
    print(json.dumps(consensus))

if __name__ == '__main__':
    main()