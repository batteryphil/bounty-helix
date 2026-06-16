import networkx as nx

def get_neighbors(graph, node):
    return list(graph.neighbors(node))

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    print(get_neighbors(G, 1))
    print(get_neighbors(G, 2))
    print(get_neighbors(G, 3))
    print(get_neighbors(G, 4))