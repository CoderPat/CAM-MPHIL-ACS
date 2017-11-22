import networkx as nx
import numpy as np
from pqdict import pqdict

def pagerank_nibble(graph, initial_vertex, connectivity, volume):
    """
    """
    alpha = connectivity

    indicator_vector = np.zeros(graph.number_of_nodes())
    indicator_vector[initial_vertex] = 1

    pagerank_vector, support_nodes = aproximate_pagerank(graph, indicator_vector, alpha, 1/(5*volume))
    
    support_nodes = sorted(support_nodes, key=lambda node: pagerank_vector[node]/graph.degree(node), reverse=True)
    print(len(support_nodes))

    optimal_sweep_set = []
    optimal_conductance = float("+inf")
    for i, node in enumerate(support_nodes):
        if pagerank_vector[node] < (1/8) * graph.degree(node) / volume:
            break
            
        cond = conductance(graph, support_nodes[:i+1])
        if cond < optimal_conductance:
            optimal_conductance = cond
            optimal_sweep_set = support_nodes[:i+1]
    
    return optimal_sweep_set, optimal_conductance

def conductance(graph, node_set):
    boundary = 0
    volume = 0
    for node in node_set:
        volume += graph.degree(node)
        boundary += len([adj_node for adj_node in graph[node].keys() if adj_node not in node_set])

    if(min(volume, 2*graph.number_of_edges() - volume) == 0):
        return float("+inf")
    
    return boundary/(min(volume, 2*graph.number_of_edges() - volume))


def aproximate_pagerank(graph, starting_vector, prob_teleport, aprox_ratio):
    p_vector = np.zeros(graph.number_of_nodes())
    r_vector = starting_vector
    non_negative_p = set()

    queue = pqdict(reverse=True)
    for node in range(0, graph.number_of_nodes()):
        queue.additem(node, r_vector[node] - aprox_ratio * graph.degree(node))

    while queue.topitem()[1] >= 0:
        node = queue.top()
        p_vector[node] += prob_teleport * r_vector[node]

        non_negative_p.add(node)
        for adj_node in graph[node].keys():
            r_vector[adj_node] +=  (1 - prob_teleport) * r_vector[node] / (2*graph.degree(node))
            queue.updateitem(adj_node, r_vector[adj_node] - aprox_ratio * graph.degree(adj_node))

        r_vector[node] = (1 - prob_teleport) * r_vector[node] / 2
        queue.updateitem(node, r_vector[node] - aprox_ratio * graph.degree(node))

    return p_vector, list(non_negative_p)

