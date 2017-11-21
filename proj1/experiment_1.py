import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pagerank_nibble import pagerank_nibble, conductance

CLUSTER_A_SIZE = 300
CLUSTER_A_MEAN_DEGREE = 60
CLUSTER_B_SIZE = 20
CLUSTER_C_SIZE = 550

def generate_random_graph(beta):
    cluster_A = nx.watts_strogatz_graph(CLUSTER_A_SIZE, CLUSTER_A_MEAN_DEGREE, beta)
    cluster_B = nx.empty_graph(CLUSTER_B_SIZE)
    cluster_C = nx.empty_graph(CLUSTER_C_SIZE)

    graph = nx.disjoint_union_all([cluster_A, cluster_B, cluster_C])

    for i in range(CLUSTER_A_SIZE + CLUSTER_B_SIZE + CLUSTER_C_SIZE):
        for j in range(i, CLUSTER_A_SIZE + CLUSTER_B_SIZE + CLUSTER_C_SIZE):
            if i < CLUSTER_A_SIZE:
                if j < CLUSTER_A_SIZE:
                    pass
                elif j < (CLUSTER_A_SIZE + CLUSTER_B_SIZE):
                    if np.random.binomial(1, 0.001):
                        graph.add_edge(i, j)
                else:
                    if np.random.binomial(1, 0.002):
                        graph.add_edge(i, j)
            elif i < (CLUSTER_A_SIZE + CLUSTER_B_SIZE):
                if j < (CLUSTER_A_SIZE + CLUSTER_B_SIZE):
                    if np.random.binomial(1, 0.3):
                        graph.add_edge(i, j)
                else:
                    if np.random.binomial(1, 0.002):
                        graph.add_edge(i, j)
            else:
                if np.random.binomial(1, 0.02):
                        graph.add_edge(i, j)


    return graph


graph = generate_random_graph(0.05)
starting_vertex = np.random.randint(0, CLUSTER_A_SIZE)
prob = 0.01
volume = 0
for i in range(CLUSTER_A_SIZE):
    volume += graph.degree(i)

print(volume)
nodes, cond = pagerank_nibble(graph, starting_vertex, prob, 10000) 

print(len(nodes))
print("Cluster Accuracy:", len([node for node in nodes if node < CLUSTER_A_SIZE])/len(nodes))
print("Cluster Ratio:", cond/conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))
print(cond, conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))

