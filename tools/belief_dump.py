import json
import pathlib
from typing import List

from helix.belief_store import belief_store

def dump_top_beliefs(top_n: int = 20) -> None:
    beliefs = belief_store.get_all()
    sorted_beliefs = sorted(beliefs, key=lambda belief: belief['confidence'], reverse=True)
    top_beliefs = sorted_beliefs[:top_n]

    with open(pathlib.Path('data/belief_snapshot.txt'), 'w') as f:
        json.dump(top_beliefs, f, indent=4)

if __name__ == '__main__':
    dump_top_beliefs()