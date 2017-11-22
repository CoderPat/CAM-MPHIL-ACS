import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('implementation')
from pagerank_nibble import pagerank_nibble, conductance


graph = nx.read_edgelist("experiments/com-amazon.ungraph.txt")
graph = nx.convert_node_labels_to_integers(graph)

starting_vertex = np.random.randint(0, graph.number_of_nodes())
nodes, cond = pagerank_nibble(graph, starting_vertex, 0.01, 9000)
print(len(nodes), cond)
