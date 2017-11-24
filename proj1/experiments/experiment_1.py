import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('implementation')
from pagerank_nibble import pagerank_nibble

CLUSTER_A_SIZE = 300
CLUSTER_A_MEAN_DEGREE = 60
CLUSTER_B_SIZE = 20
CLUSTER_C_SIZE = 550

DEBUG = True

def debug(message):
    if DEBUG:
        print(message)

def generate_random_clustered_graph(beta):
    '''
    Generates a random graph according to the description in the paper.

    
    '''
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



#Test for variations in beta with fixed alpha
accuracys = []
ratios = []
sizes = []
betas = np.arange(0.01, 0.11, 0.01)
for beta in betas:
    debug("running with beta: %s" % beta)

    graph = generate_random_clustered_graph(beta)
    starting_vertex = np.random.randint(0, CLUSTER_A_SIZE)
    nodes, cond = pagerank_nibble(graph, starting_vertex, 0.005, 10000)
    
    accuracys.append(len([node for node in nodes if node < CLUSTER_A_SIZE])/len(nodes))
    ratios.append(cond/nx.conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))

plt.figure(1)
plt.plot(betas, accuracys)
plt.figure(2)
plt.plot(betas, ratios)
plt.show()

#Tests for variations in alpha with fixed beta
accuracys = []
ratios = []
sizes = []
alphas = np.arange(0.001, 0.031, 0.005)
graph = generate_random_clustered_graph(beta)
for alpha in alphas:
    debug("running with alpha: %s" % alpha)
    starting_vertex = np.random.randint(0, CLUSTER_A_SIZE)
    nodes, cond = pagerank_nibble(graph, starting_vertex, alpha, 10000)
    
    accuracys.append(len([node for node in nodes if node < CLUSTER_A_SIZE])/len(nodes))
    ratios.append(cond/nx.conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))

plt.figure(1)
plt.plot(alphas, accuracys)
plt.figure(2)
plt.plot(alphas, ratios)
plt.show()