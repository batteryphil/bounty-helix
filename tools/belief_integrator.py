import json
from typing import List, Dict

def integrate_beliefs(conflicting_beliefs: List[Dict]) -> Dict:
    # Calculate the average confidence score for each belief
    belief_scores = {belief['statement']: belief['confidence'] for belief in conflicting_beliefs}
    avg_confidence = sum(belief_scores.values()) / len(conflicting_beliefs)

    # Determine the belief with the highest confidence score
    most_confident_belief = max(conflicting_beliefs, key=lambda belief: belief['confidence'])

    # Update the belief with the average confidence score
    most_confident_belief['confidence'] = avg_confidence

    # Create a new belief with the integrated statement and average confidence
    integrated_belief = {
        'statement': ' '.join(belief['statement'] for belief in conflicting_beliefs),
        'confidence': avg_confidence
    }

    return integrated_belief

def main():
    # Example usage
    conflicting_beliefs = [
        {'statement': 'The sky is blue', 'confidence': 0.8},
        {'statement': 'The sky is green', 'confidence': 0.2},
        {'statement': 'The sky is purple', 'confidence': 0.3}
    ]

    integrated_belief = integrate_beliefs(conflicting_beliefs)
    print(json.dumps(integrated_belief, indent=2))

if __name__ == '__main__':
    main()