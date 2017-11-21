import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../implementation')
from pagerank_nibble import pagerank_nibble, conductance

CLUSTER_A_SIZE = 300
CLUSTER_A_MEAN_DEGREE = 60
CLUSTER_B_SIZE = 20
CLUSTER_C_SIZE = 550

DEBUG = True

def debug(message):
    if DEBUG:
        print(message)

def generate_random_clustered_graph(beta):
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




accuracys = []
ratios = []
sizes = []
betas = np.arange(0.01, 0.11, 0.025)
for beta in betas:
    debug("running with beta: %s" % beta)

    graph = generate_random_clustered_graph(beta)
    starting_vertex = np.random.randint(0, CLUSTER_A_SIZE)

    
    nodes, cond = pagerank_nibble(graph, starting_vertex, 0.1*beta, 90000) 
    accuracys.append(len([node for node in nodes if node < CLUSTER_A_SIZE])/len(nodes))
    sizes.append(len(nodes))
    ratios.append(cond/conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))

plt.figure(1)
plt.plot(betas, accuracys)
plt.show()
plt.figure(2)
plt.plot(betas, ratios)
plt.show()

print(sizes)