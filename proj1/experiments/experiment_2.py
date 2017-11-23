import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
sys.path.append('implementation')
from pagerank_nibble import pagerank_nibble

#Extracts a networkx graph from a file
def get_graph(path):
    graph = nx.read_edgelist(path)
    return nx.convert_node_labels_to_integers(graph)

#Extracts the previously analysed communities for a graph from a file
def get_communities(path):
    node_communities = defaultdict(list)
    with open(path, 'r') as comm_file: 
        for i, line in enumerate(comm_file):
            for node in line.split():
                node_communities[int(node)-1].append(i)
    return node_communities

#Given a cluster and a communities dictionary, returns the number of nodes in each community in the cluster
def get_cluster_communities(node_list, communities):
    cluster_communities = defaultdict(lambda: 0)
    for node in node_list:
        for community in communities[node]:
            cluster_communities[community] += 1
    return cluster_communities


graph = get_graph("experiments/com-amazon.ungraph.txt")
communities =  get_communities("experiments/com-amazon.all.dedup.cmty.txt")

starting_vertex = 9649 #np.random.randint(0, graph.number_of_nodes())
nodes, cond = pagerank_nibble(graph, starting_vertex, 0.01, 2000)
print(len(nodes), cond)

cluster_communities = get_cluster_communities(nodes, communities)
cluster_communities = sorted(cluster_communities.values(), reverse=True)

for cluster in cluster_communities:
    print(cluster/len(nodes))

