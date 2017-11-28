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



"""
#Test for variations in beta with fixed alpha
betas = np.arange(0.01, 0.11, 0.01)
tot_accuracies = []
tot_ratios = []
for _ in range(10):
    accuracies = []
    ratios = []
    for beta in betas:
        debug("running with beta: %s" % beta)

        graph = generate_random_clustered_graph(beta)
        starting_vertex = np.random.randint(0, CLUSTER_A_SIZE)
        nodes, cond = pagerank_nibble(graph, starting_vertex, 0.005, 10000)
    
        accuracies.append(len([node for node in nodes if node < CLUSTER_A_SIZE])/len(nodes))
        ratios.append(cond/nx.conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))
    tot_accuracies.append(accuracies)
    tot_ratios.append(ratios)

accuracies = np.mean(np.array(tot_accuracies), axis=0)
ratios = np.mean(np.array(tot_ratios), axis=0)

plt.plot(betas, accuracies, color='red')
plt.savefig("betas-accuracy.png")
plt.clf()
plt.plot(betas, ratios)
plt.savefig("betas-ratio.png")
plt.clf()
"""

#Tests for variations in starting vertex
accuracies1 = []
accuracies2 = []
ratios1 = []
ratios2 = []
for _ in range(10):
    print("a")
    graph = generate_random_clustered_graph(0.05)
    max_outer_edges = 0
    max_inner_edges = 0
    for node in range(CLUSTER_A_SIZE):
        inner_edges = 0
        outer_edges = 0
        for adj_node in graph[node].keys():
            if adj_node < CLUSTER_A_SIZE:
                 inner_edges += 1
            else:
                outer_edges += 1
        if inner_edges > max_inner_edges:
            max_inner_edges = inner_edges
            inner_node = node
        if outer_edges > max_outer_edges:
            max_outer_edges = outer_edges
            outer_node = node

    nodes1, cond1 = pagerank_nibble(graph, inner_node, 0.005, 10000)
    nodes2, cond2 = pagerank_nibble(graph, outer_node, 0.005, 10000)

    accuracies1.append(len([node for node in nodes1 if node < CLUSTER_A_SIZE])/len(nodes1))
    ratios1.append(cond1/nx.conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))
        
    accuracies2.append(len([node for node in nodes2 if node < CLUSTER_A_SIZE])/len(nodes2))
    ratios2.append(cond2/nx.conductance(graph, [i for i in range(CLUSTER_A_SIZE)]))

accuracy1 = np.mean(np.array(accuracies1))
ratio1 = np.mean(np.array(ratios1))

accuracy2 = np.mean(np.array(accuracies2))
ratio2 = np.mean(np.array(ratios2))

ind = np.arange(1, 3)

plt.bar(ind, (accuracy1, accuracy2), 0.5)
plt.savefig("vertex-accuracy.png")
plt.clf()

plt.bar(ind, (ratio1, ratio2), 0.5, color='red')
plt.savefig("vertex-ratio.png")
plt.clf()
    


    

