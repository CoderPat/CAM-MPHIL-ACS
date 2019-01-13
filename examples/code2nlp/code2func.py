"""
Usage:
    code2func.py [options]

Options:
    -h --help                       Show this screen.
    --log_dir DIR                   Log dir name.
    --data-dir DIR                  Data dir name.
    --restore FILE                  File to restore weights from.
    --no-train                      Sets epochs to zero (only for already trained models)
    --freeze-graph-model            Freeze weights of graph model components.
    --training-data FILE            Location of the training data
    --validation-data FILE          Location of the validation data
    --testing-data FILE             Location of the test data
    --save-alignments FILE,FILE     Locations of input and output vectorizers       
"""

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import random
import os
import json
import sys, traceback
import pdb

from graph2sequence.graph2seq import Graph2Seq

from graph2sequence.base.utils import compute_bleu, compute_f1

MAX_VERTICES_GRAPH = 700

CONFIG = {
    'batch_size': 50000,
    'graph_state_dropout_keep_prob': 0.7,
    'decoder_cells_dropout_keep_prob': 0.9,
    'learning_rate': 0.001,
    'layer_timesteps': [2, 2, 2, 2],
    'attention_scope': None,
    'random_seed' : None,
    'beam_width': 1,
}

def scope_labels(node_labels, node_types, scopes):
    if scopes is None:
        return node_labels
    
    scoped_all_labels = []
    for labels, types in zip(node_labels, node_types):
        scoped_labels = []
        for label, t in zip(labels, types):
            if (type(scopes) == list and t in scopes) or \
               (type(scopes) == int and t == scopes):
                scoped_labels.append(label)
        scoped_all_labels.append(scoped_labels)
    return scoped_all_labels

def load_data(data_dir, file_name, restrict = None):
    full_path = os.path.join(data_dir, file_name)

    print("Loading data from %s" % full_path)
    with open(full_path, 'r') as f:
        raw_data = json.load(f)

    new_data = []
    for d in raw_data:
        g = {}
        g["graph"] = d["graph"]
        (data, indices, indptr, shape) = d["node_features"]
        g["node_types"] = np.array(d["node_types"])
        g["node_features"] = csr_matrix((data, indices, indptr), shape)
        g["output"] = np.array(d["output_features"])

        if shape[0] > MAX_VERTICES_GRAPH or len(g["output"]) == 0:
            continue
        
        if len(d["graph"]) == 0:
            continue

        new_data.append(g)

    print("Using %d out of %d" % (len(new_data), len(raw_data)))
    return new_data

def save_alignments(coefs, codes, outputs, references, path):
    data = {"src_labels": codes,
            "tgt_labels": outputs,
            "references": references,
            "coefs": []}
    with open(path, "wb") as f:
        pickle.dump(data,f, pickle.HIGHEST_PROTOCOL)

def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir') or 'dataset/processed_data/'
    train_data = args.get('--training-data') or "graphs-body-name-train.json"
    valid_data = args.get('--validation-data') or "graphs-body-name-valid.json"
    test_data = args.get('--testing-data') or "graphs-body-name-test.json"
    vects = args.get('--save-alignments')

    train_data = load_data(data_dir, train_data)
    valid_data = load_data(data_dir, valid_data)
    test_data = load_data(data_dir, test_data)

    if args.get('--no-train'):
        CONFIG['num_epochs'] = 0
        
    args['--config'] =  json.dumps(CONFIG)

    try:
        model = Graph2Seq(args)
        model.train(train_data, valid_data)
        model.evaluate(test_data)
        
        if vects is not None:
            input_vect, output_vect = vects.split(',') 

            print("Loading output vectorizer")
            with open(output_vect, 'rb') as f:
                output_vect = pickle.load(f)
                print(len(output_vect._Vectorizer__reversed_dictionary))

            print("Loading input vectorizer")
            with open(input_vect, 'rb') as f:
                input_vect = pickle.load(f)

            outputs, coefs = model.infer(test_data, alignment_history=True)
            outputs = [output_vect.devectorize(output, subtokens=False) for output in outputs]
            codes = scope_labels(input_vect.devectorize([d["node_features"] for d in test_data], subtokens=True), [d["node_types"] for d in test_data], CONFIG['attention_scope'])
            references = output_vect.devectorize([g['output'] for g in test_data], subtokens=False)

            save_alignments(coefs, codes, outputs, references, "attention_maps_body_name.pickle")
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    main()

