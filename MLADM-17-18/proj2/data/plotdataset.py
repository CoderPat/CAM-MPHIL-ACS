import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np

from utils import load_data

def connected_subgraph(graph, n_nodes):
    node = np.random.choice(graph.nodes())
    visited = []
    while len(visited) < n_nodes:
        neighbors = list(graph[node].keys())
        node = np.random.choice(neighbors)
        if node not in visited:
            visited.append(node)
    return visited

def plot_subgraph(adj, labels):
    graph = nx.to_networkx_graph(adj)
    node_list = connected_subgraph(graph, 20)
    subgraph = graph.subgraph(node_list)
    num_labels = np.max(labels) + 1
    nx.draw(subgraph, cmap=plt.get_cmap('jet'), node_color=labels[node_list])
    plt.show()


