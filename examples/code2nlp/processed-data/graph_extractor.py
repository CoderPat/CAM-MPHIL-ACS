#!/usr/bin/env/python
"""
Usage:
    graph_extractor.py [options]

Options:
    -h --help                Show this screen.
    --code_data FILE         Prefix of the code data location 
    --label_data FILE        Prefix of the label data location 
    --split TYPE             The split suffix of the code and label data (train/valid/test)
    --load-input-vect FILE   File to load a previously created vectorizer for encoding token labels.
    --save-input-vect FILE   File to save the created vectorizer for encoding token labels.
    --load-output-vect FILE  File to load a previously created vectorizer for encoding code descriptions.
    --save-output-vect FILE  File to save the created vectorizer for encoding code descriptions.
"""
from ast import *
from re import finditer
from collections import defaultdict
from docopt import docopt
import gc
import json
import itertools
import pickle


import networkx as nx
from networkx.drawing.nx_agraph import write_dot

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from ast_graph_generator import AstGraphGenerator

def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    subtoken_dict = defaultdict(lambda: 0)
    for subtoken in [m.group(0).lower() for m in matches]:
        subtoken_dict[subtoken] += 1
    return dict(subtoken_dict)

def plot_code_graph(snippet):
    print(snippet)
    visitor = AstGraphGenerator() 
    visitor.visit(parse(snippet))
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

def process_data(functs, docs, input_vectorizer, output_vectorizer):
    data, graph_node_labels, docs_words = ([], [], [])
    num_inits, errors = (0,0)
    tokenizer = output_vectorizer.build_tokenizer() if output_vectorizer is not None else \
                CountVectorizer().build_tokenizer()
    for idx, (funct, doc) in enumerate(zip(functs, docs)):
        try:
            visitor = AstGraphGenerator() 
            visitor.visit(parse(funct))

            edge_list = [(origin, t, destination) for (origin, destination), edges in visitor.graph.items() for t in edges]

            docs_words.append(tokenizer(doc))

            graph_node_labels.append([label for _, label in sorted(visitor.node_label.items())])
            data.append({"graph":edge_list})
        except:
            errors += 1

    print("Generated %d graphs out of %d snippets" % (len(functions) - errors, len(functions)))

    all_node_labels = [camel_case_split(label) for label in itertools.chain.from_iterable(graph_node_labels)]
    all_words = list(itertools.chain.from_iterable(docs_words))

    if input_vectorizer is None:
        input_vectorizer = DictVectorizer()
        all_node_counts = input_vectorizer.fit_transform(all_node_labels)
    else:
        all_node_counts = input_vectorizer.transform(all_node_labels)

    if output_vectorizer is None:
        output_vectorizer = CountVectorizer()
        all_word_indices = output_vectorizer.fit_transform(all_words).indices
    else:
        all_word_indices = output_vectorizer.transform(all_words).indices

    prev_ind = 0
    for i, labels in enumerate(graph_node_labels):
        counts = all_node_counts[prev_ind:prev_ind+len(labels), :]
        data[i]["node_features"] = (counts.data.tolist(), counts.indices.tolist(), 
                                    counts.indptr.tolist(), counts.shape)
        prev_ind += len(labels)

    prev_ind = 0
    for i, sentence in enumerate(docs_words):
        data[i]["output_features"] = all_word_indices[prev_ind:prev_ind+len(sentence)].tolist()
        prev_ind += len(sentence)

    return data, input_vectorizer, output_vectorizer


if __name__ == "__main__":
    args = docopt(__doc__)
    code_data = args.get('--code_data') or "../data/V2/repo_split/repo_split.parallel_methods_declbodies"
    label_data = args.get('--label_data') or "../data/V2/repo_split/repo_split.parallel_methods_desc"
    split = args.get('--split') or "train"
    output_file = args.get('--output_file') or "graphs-" + split + ".json"
    input_vectorizer = args.get('--load-input-vec')
    save_input_vect = args.get('--save-input-vec')
    output_vectorizer = args.get('--load-output-vec')
    save_output_vect = args.get('--save-output-vec')
    print_example = args.get('--print-example')

    with open(code_data + "." + split) as f:
        functions = f.readlines()
    with open(label_data + "." + split, errors='ignore') as f:
        docs = f.readlines()

    functions = [function.replace("DCNL ", "\n").replace("DCSP ", "\t") for function in functions]
    docs = [doc.replace("DCNL ", "\n").replace("DCSP ", "\t") for doc in docs]

    if input_vectorizer is not None:
        with open(input_vectorizer, 'rb') as f:
            input_vectorizer = pickle.load(f)

    if output_vectorizer is not None:
        with open(output_vectorizer, 'rb') as f:
            output_vectorizer = pickle.load(f)
    
    data, input_vectorizer, output_vectorizer = process_data(functions, docs, input_vectorizer, output_vectorizer)

    if save_input_vect is not None:
        with open(save_input_vect, 'wb') as f:
            pickle.dump(input_vectorizer, f)

    if save_output_vect is not None:
        with open(save_output_vect, 'wb') as f:
            pickle.dump(output_vectorizer, f)

    with open(output_file, "w") as f:
        json.dump(data, f)
    
"""

"""
