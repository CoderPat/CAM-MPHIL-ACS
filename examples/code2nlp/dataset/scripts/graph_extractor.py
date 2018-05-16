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
"""
from ast import parse
import re
from collections import defaultdict
from docopt import docopt
import gc
import json
import itertools
import pickle
from tempfile import NamedTemporaryFile
import subprocess

from scipy.sparse import csr_matrix
from nltk import word_tokenize

import numpy as np
import matplotlib.pyplot as plt

from graph2sequence.base.utils import Vectorizer
from ast_graph_generator import *
from sklearn.feature_extraction.text import CountVectorizer

MOSES_TOKENIZER = False

def moses_tonenization(descriptions):
    temp_file = NamedTemporaryFile(delete=True)
    with open(temp_file.name, 'w') as g:
        g.writelines(descriptions)

    with open(temp_file.name, 'r') as g:
        process = subprocess.run(["perl", "tokenizer.perl"], 
                                  stdout=subprocess.PIPE,
                                  stdin=temp_file)
    return process.stdout.decode('utf-8').split('\n')[:-1]


#Tokenizes code identifiers
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

#Extracts function names
def decl_tokenizer(decl):
    function_name = re.search('(?<=def )[\w_-]+(?=\(.*\):)', decl).group(0)
    return splitter(function_name, ordered=True)


def process_data(inputs, outputs, task_type, input_vectorizer, output_vectorizer):
    data, graph_node_labels, docs_words = ([], [], [])
    num_inits, errors = (0,0)
    doc_tokenizer = (lambda s: s.split()) if MOSES_TOKENIZER else word_tokenize
    
    for idx, (inp, output) in enumerate(zip(inputs, outputs)):
        try:
            visitor = AstGraphGenerator() 
            visitor.visit(parse(inp))

            edge_list = [(origin, t, destination) for (origin, destination), edges in visitor.graph.items() for t in edges]

            if task_type == "func-doc":
                docs_words.append(doc_tokenizer(output))
            if task_type == "body-decl":
                docs_words.append(decl_tokenizer(output))

            graph_node_labels.append([label.strip() for _, label in sorted(visitor.node_label.items())])
            data.append({"graph":edge_list, 
                         "node_types": [node_type for _, node_type in sorted(visitor.node_type.items())],
                         "primary_path": visitor.terminal_path})


        except Exception as e:
            errors += 1

    print("Generated %d graphs out of %d snippets" % (len(inputs) - errors, len(inputs)))

    node_labels = [[splitter(label) for label in graph] for graph in graph_node_labels]

    if input_vectorizer is None:
        input_vectorizer = Vectorizer()
        node_counts = input_vectorizer.build_and_vectorize(node_labels)
    else:
        node_counts = input_vectorizer.vectorize(node_labels)

    if output_vectorizer is None:
        output_vectorizer = Vectorizer()
        word_indices = output_vectorizer.build_and_vectorize(docs_words, subtokens=False)
    else:
        word_indices = output_vectorizer.vectorize(docs_words, subtokens=False)


    for i, (graph_counts, graph_indices) in enumerate(zip(node_counts, word_indices)):
        data[i]["node_features"] = (graph_counts.data.tolist(), graph_counts.indices.tolist(), 
                                    graph_counts.indptr.tolist(), graph_counts.shape)

        data[i]["output_features"] = graph_indices

    return data, input_vectorizer, output_vectorizer


if __name__ == "__main__":
    args = docopt(__doc__)
    task_type = args.get('--task_type') or "func-doc"
    split = args.get('--split') or "train"

    #Change default files based on the task
    if task_type == 'func-doc':
        code_data = args.get('--code_data') or "../cleaned_data/parallel_declbodies"
        label_data = args.get('--label_data') or "../cleaned_data/parallel_desc"
    elif task_type == 'body-name':
        code_data = args.get('--code_data') or "../cleaned_data/parallel_bodies"
        label_data = args.get('--label_data') or "../cleaned_data/parallel_decl"

    #Change default vectorizer behaviour based on split
    if split == "train":
        input_vectorizer = args.get('--load-input-vect') 
        save_input_vect = args.get('--save-input-vect') or "../processed_data/input-" + task_type + "-vect.pickle"
        output_vectorizer = args.get('--load-output-vect')
        save_output_vect = args.get('--save-output-vect') or "../processed_data/output-" + task_type + "-vect.pickle"
    else:
        input_vectorizer = args.get('--load-input-vect') or "../processed_data/input-" + task_type + "-vect.pickle"
        save_input_vect = args.get('--save-input-vect') 
        output_vectorizer = args.get('--load-output-vect') or "../processed_data/output-" + task_type + "-vect.pickle"
        save_output_vect = args.get('--save-output-vect') 

    output_file = args.get('--output_file') or "../processed_data/graphs-" + task_type + "-" + split + ".json"

    with open(code_data + "." + split) as f:
        inputs = [line for line in f.readlines()]
    with open(label_data + "." + split, 'rb') as f:
        labels = [line.decode('utf-8') for line in f.readlines()]

    inputs = [inp.replace("DCNL ", "\n").replace("DCSP ", "\t") for inp in inputs]

    #Unident body so it can be parsed
    if task_type == 'body-decl':
        inputs = ["\n".join([line[1:] for line in inp.split("\n")]) for inp in inputs]

    if MOSES_TOKENIZER:
        labels = moses_tonenization(labels)

    labels = [label.replace("DCNL ", "\n").replace("DCSP ", "\t") for label in labels]

    if input_vectorizer is not None:
        print("Loading input vectorizer")
        with open(input_vectorizer, 'rb') as f:
            input_vectorizer = pickle.load(f)

    if output_vectorizer is not None:
        print("Loading output vectorizer")
        with open(output_vectorizer, 'rb') as f:
            output_vectorizer = pickle.load(f)
    
    data, input_vectorizer, output_vectorizer = process_data(
        inputs, labels, task_type, input_vectorizer, output_vectorizer)

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
