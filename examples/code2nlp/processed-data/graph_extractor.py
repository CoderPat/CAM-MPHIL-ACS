from ast import *
from re import finditer
from collections import defaultdict
import gc
import json
import itertools

import networkx as nx
from networkx.drawing.nx_agraph import write_dot

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer

from ast_graph_generator import AstGraphGenerator

print_random_example = False

def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    subtoken_dict = defaultdict(lambda: 0)
    for subtoken in [m.group(0).lower() for m in matches]:
        subtoken_dict[subtoken] += 1
    return dict(subtoken_dict)

def process_data(bodies, decls):
    data = []
    graph_node_labels = []
    num_inits = 0
    for idx, (body, decl) in enumerate(zip(bodies, decls)):
        try:
            visitor = AstGraphGenerator() 
            visitor.visit(parse(body))
            edge_list = [(origin, t, destination) for (origin, destination), edges in visitor.graph.items() for t in edges]
            label = "__init__" in decl
            num_inits += 1 if label else 0
            graph_node_labels.append([label for _, label in sorted(visitor.node_label.items())])
            data.append({"graph":edge_list, "label":label})
        except:
            errors += 1
        print("Generated %d graphs out of %d snippets" % (len(bodies) - errors, len(bodies)))
    return data

if __name__ == "__main__":
    
    with open("../data/V2/repo_split/repo_split.parallel_methods_bodies.train") as f:
        valid_bodies = f.readlines()
    with open("../data/V2/repo_split/repo_split.parallel_methods_decl.train") as f:
        valid_decls = f.readlines()

    bodies = [body.replace("DCNL ", "\n").replace("DCSP ", "\t") for body in bodies]
    bodies = ["\n".join([line[1:] for line in body.split("\n")]) for body in bodies]
    errors = 0

    print("Generated %d graphs out of %d snippets" % (len(bodies) - errors, len(bodies)))
    all_node_labels = [camel_case_split(label) for label in itertools.chain.from_iterable(graph_node_labels)]
    vectorizer = DictVectorizer()
    all_node_counts = vectorizer.fit_transform(all_node_labels)

    node_counts = []
    prev_ind = 0
    for i, labels in enumerate(graph_node_labels):
        counts = all_node_counts[prev_ind:prev_ind+len(labels), :]
        data[i]["node_features"] = (counts.data.tolist(), counts.indices.tolist(), 
                                    counts.indptr.tolist(), counts.shape)
        prev_ind += len(labels)


    new_data = []
    for graph_data in data:
        if graph_data['label'] == 0:
            if num_inits <= 0:
                continue
            new_data.append(graph_data)
            num_inits -= 0.75
        else:
            new_data.append(graph_data)

    print(len(new_data))
    with open("graphs.json", "w") as f:
        json.dump(new_data, f)
    
"""
    if print_random_example:
        print(bodies[412])
        visitor = AstGraphGenerator() 
        visitor.visit(parse(bodies[412]))
        graph = nx.parse_edgelist(["%d %d {'type': %d}" 
                    % (origin, destination, t) for (origin, destination), edges in visitor.graph.items() for t in edges], 
                                    nodetype = int,
                                    create_using = nx.DiGraph())
        pos=nx.nx_agraph.graphviz_layout(graph, prog='dot')

        
        edges = graph.edges()
        colors = ['green' if graph[source][destination]['type'] else 'blue' for source, destination in edges]
        nx.draw(graph, pos, with_labels=False, arrows=True, edges=edges, edge_color=colors)
        nx.draw_networkx_labels(graph, pos, labels=visitor.node_label)
        plt.draw()
        plt.show()
"""