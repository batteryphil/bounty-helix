import json
import networkx as nx
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with file_path.open('r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def jsonl_to_nxgraph(jsonl_data):
    G = nx.DiGraph()
    for item in jsonl_data:
        G.add_node(item['id'])
        for neighbor in item['neighbors']:
            G.add_edge(item['id'], neighbor)
    return G

def get_neighbors(G, node_id):
    return list(G.predecessors(node_id))

def main():
    jsonl_file_path = Path('data.jsonl')
    jsonl_data = load_jsonl(jsonl_file_path)
    G = jsonl_to_nxgraph(jsonl_data)
    node_id = 'node1'
    neighbors = get_neighbors(G, node_id)
    print(f"Neighbors of {node_id}: {neighbors}")

if __name__ == '__main__':
    main()