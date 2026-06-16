import json
import os
from pathlib import Path
from typing import List, Dict

def get_all_beliefs() -> Dict[str, Dict[str, float]]:
    # Simulated function to get all beliefs
    # In a real implementation, this would query the belief store
    return {
        'belief1': {'confidence': 0.95},
        'belief2': {'confidence': 0.87},
        'belief3': {'confidence': 0.42},
        'belief4': {'confidence': 0.68},
        'belief5': {'confidence': 0.12},
        'belief6': {'confidence': 0.91},
        'belief7': {'confidence': 0.03},
        'belief8': {'confidence': 0.56},
        'belief9': {'confidence': 0.78},
        'belief10': {'confidence': 0.60},
        'belief11': {'confidence': 0.88},
        'belief12': {'confidence': 0.19},
        'belief13': {'confidence': 0.45},
    }