import networkx as nx
import numpy as np

from collections import defaultdict
from pqdict import pqdict

def pagerank_nibble(graph, initial_vertex, alpha, target_volume):
    """ An local clustering algorithm based on the paper https://arxiv.org/abs/1304.8132

    Args:
        graph (NetworkX graph) : the graph where we want to cluster
        initial_vertex (int) : a node situated in the objective cluster
        connectivity (float) : an estimate of the internal connectivity of the objective cluster
        target_volume (int) : an estimate of the volume of the objective cluster

    Returns:
        list int : the list of the nodes in the computed cluster
        int : the conductance of the computed cluster
    """
    indicator_vector = defaultdict(lambda: 0)
    indicator_vector[initial_vertex] = 1

    pagerank_vector = aproximate_pagerank(graph, indicator_vector, alpha, 1/(10*target_volume))
    support_nodes = list(pagerank_vector.keys())

    support_nodes = sorted(support_nodes, key=lambda node: pagerank_vector[node]/graph.degree(node), reverse=True)

    optimal_sweep_set = []
    optimal_conductance = float("+inf")
    volume = 0 
    boundary_size = 0
    for i, node in enumerate(support_nodes):

        #Calculate volume and boundary for the new vertices set using the previous values
        volume += graph.degree(node)
        for adj_node in graph[node].keys():
            if adj_node in support_nodes[:i]:
                #If this edge exists it means that it is already in the boundary and needs to be removed 
                boundary_size -= 1
            else:
                boundary_size += 1

        #Breaking condition for the loop
        if pagerank_vector[node] < (1/8) * graph.degree(node) / target_volume:
            break
            
        #Calculate conductance (we are not interested in empty or full-graph sized clusters)
        if volume == 0 or volume == 2*graph.number_of_edges():
            cond = float("+inf")
        else:
            cond = boundary_size/min(volume, 2*graph.number_of_edges() - volume)

        #Check if its the best
        if cond < optimal_conductance:
            optimal_conductance = cond
            optimal_sweep_set = support_nodes[:i+1]
    
    return optimal_sweep_set, optimal_conductance


def aproximate_pagerank(graph, starting_vector, prob_teleport, aprox_ratio):
    """ An algorithm to calculate the e-aproximate pagerank vector, as formulated in https://arxiv.org/abs/1304.8132
    
    Args:
        graph (NetworkX graph) : the graph where we want to cluster
        starting_vector ((int, float) defaultdict): a (lazy) vector of the initial distribution of the nodes
        prob_teleport (float) : the teleport probability (informally the probability of randomly going back to the origin)
        aprox_ration (float) : an estimate of the volume of the objective cluster

    Returns:
        (int, float) defaultdict  : the aproximate pagerank (lazy) vector
    """
    #A "lazy vector" is used to avoid having to initialize a full vector, which would be linear in the graph size
    p_vector = defaultdict(lambda: 0)
    r_vector = starting_vector

    #A priority queue (heap) provides an effecient way to obtain valid nodes push
    queue = pqdict(reverse=True)
    for node,r_value in r_vector.items():
        queue.additem(node, r_value - aprox_ratio * graph.degree(node))

    while queue.topitem()[1] >= 0:
        node = queue.top()
        p_vector[node] += prob_teleport * r_vector[node]

        for adj_node in graph[node].keys():
            r_vector[adj_node] +=  (1 - prob_teleport) * r_vector[node] / (2*graph.degree(node))
            if adj_node not in queue:
                queue.additem(adj_node, r_vector[adj_node] - aprox_ratio * graph.degree(adj_node))
            else:
                queue.updateitem(adj_node, r_vector[adj_node] - aprox_ratio * graph.degree(adj_node))

        r_vector[node] = (1 - prob_teleport) * r_vector[node] / 2
        queue.updateitem(node, r_vector[node] - aprox_ratio * graph.degree(node))

    return p_vector

