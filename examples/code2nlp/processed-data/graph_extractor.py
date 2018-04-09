#!/usr/bin/env/python
"""
Usage:
    graph_extractor.py [options]

Options:
    -h --help                Show this screen.
    --code_data FILE         Prefix of the code data location 
    --label_data FILE        Prefix of the label data location 
    --task_type TYPE         Type of the task [func-doc|body-decl]
    --split TYPE             The split suffix of the code and label data [train|valid|test]
    --load-input-vect FILE   File to load a previously created vectorizer for encoding token labels.
    --save-input-vect FILE   File to save the created vectorizer for encoding token labels.
    --load-output-vect FILE  File to load a previously created vectorizer for encoding code descriptions.
    --save-output-vect FILE  File to save the created vectorizer for encoding code descriptions.
    --print-example          Prints random graph examples
"""
from ast import *
import re
from collections import defaultdict
from docopt import docopt
import gc
import json
import itertools
import pickle

from scipy.sparse import csr_matrix

from pygraphviz import *

import numpy as np
import matplotlib.pyplot as plt

from graph2sequence.base.utils import Vectorizer
from ast_graph_generator import *
from sklearn.feature_extraction.text import CountVectorizer


def splitter(identifier, ordered=False):
    identifiers = re.split('_|-', identifier)
    subtoken_dict = defaultdict(lambda: 0)
    subtoken_list = []
    for identifier in identifiers:
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        for subtoken in [m.group(0).lower() for m in matches]:
            if ordered:
                subtoken_list.append(subtoken)
            else:
                subtoken_dict[subtoken] += 1
            
    return subtoken_list if ordered else dict(subtoken_dict)

def decl_tokenizer(decl):
    function_name = re.search('(?<=def )[\w_-]+(?=\(.*\):)', decl).group(0)
    return splitter(function_name, ordered=True)

def plot_code_graph(snippet):

    print(snippet)

    color_dict = {
        0 : 'blue',
        1 : 'yellow',
        2 : 'black',
        3 : 'red',
        4 : 'green',
        5 : 'brown',
    }
    visitor = AstGraphGenerator() 
    visitor.visit(parse(snippet))

    A = AGraph(strict=False, directed=True)

    for (u, v), edges in visitor.graph.items():
        for edge in edges:
            A.add_edge(u, v, color=color_dict[edge])
    
    for i, (nid, label) in enumerate(visitor.node_label.items()):
        A.get_node(i).attr['label'] = "%s (%d)" % (label, nid)
    

    A.layout('dot')                                                                 
    A.draw('multi.png')   


def process_data(inputs, outputs, task_type, input_vectorizer, output_vectorizer):
    data, graph_node_labels, docs_words = ([], [], [])
    num_inits, errors = (0,0)
    doc_tokenizer = CountVectorizer().build_tokenizer()
    
    for idx, (inp, output) in enumerate(zip(inputs, outputs)):
        try:
            visitor = AstGraphGenerator() 
            visitor.visit(parse(inp))

            edge_list = [(origin, t, destination) for (origin, destination), edges in visitor.graph.items() for t in edges]

            if task_type == "func-doc":
                docs_words.append(doc_tokenizer(output))
            if task_type == "body-decl":
                docs_words.append(decl_tokenizer(output))

            graph_node_labels.append([label for _, label in sorted(visitor.node_label.items())])
            data.append({"graph":edge_list})

        except Exception as e:
            errors += 1

    print("Generated %d graphs out of %d snippets" % (len(inputs) - errors, len(inputs)))

    all_node_labels = [splitter(label) for label in itertools.chain.from_iterable(graph_node_labels)]
    all_words = [{word: 1} for word in itertools.chain.from_iterable(docs_words)]

    if input_vectorizer is None:
        input_vectorizer = Vectorizer()
        all_node_counts = input_vectorizer.build_and_vectorize(all_node_labels)
    else:
        all_node_counts = input_vectorizer.vectorize(all_node_labels)

    if output_vectorizer is None:
        output_vectorizer = Vectorizer()
        all_word_indices = output_vectorizer.build_and_vectorize(all_words).indices
    else:
        all_word_indices = output_vectorizer.vectorize(all_words).indices

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
    task_type = args.get('--task_type') or "func-doc"
    split = args.get('--split') or "train"
    output_file = args.get('--output_file') or "graphs-" + task_type + "-" + split + ".json"
    input_vectorizer = args.get('--load-input-vect')
    save_input_vect = args.get('--save-input-vect')
    output_vectorizer = args.get('--load-output-vect')
    save_output_vect = args.get('--save-output-vect')
    print_example = args.get('--print-example')


    with open(code_data + "." + split) as f:
        inputs = f.readlines()
    with open(label_data + "." + split, errors='ignore') as f:
        labels = f.readlines()

    inputs = [inp.replace("DCNL ", "\n").replace("DCSP ", "\t") for inp in inputs]
    if task_type == 'body-decl':
        inputs = ["\n".join([line[1:] for line in inp.split("\n")]) for inp in inputs]

    labels = [label.replace("DCNL ", "\n").replace("DCSP ", "\t") for label in labels]

    if input_vectorizer is not None:
        print("Loading input vectorizer")
        with open(input_vectorizer, 'rb') as f:
            input_vectorizer = pickle.load(f)

    if output_vectorizer is not None:
        print("Loading output vectorizer")
        with open(output_vectorizer, 'rb') as f:
            output_vectorizer = pickle.load(f)
    
    data, input_vectorizer, output_vectorizer = process_data(inputs, labels, task_type, 
                                                             input_vectorizer, output_vectorizer)

    if save_input_vect is not None:
        print("Saving input vectorizer")
        with open(save_input_vect, 'wb') as f:
            pickle.dump(input_vectorizer, f)

    if save_output_vect is not None:
        print("Saving output vectorizer")
        with open(save_output_vect, 'wb') as f:
            pickle.dump(output_vectorizer, f)

    with open(output_file, "w") as f:
        json.dump(data, f)
