"""
Usage:
    code2init.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --training-data FILE     Location of the training data
    --validation-data FILE   Location of the training data
"""

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle

import numpy as np

import os
import json
import sys, traceback
import pdb

from graph2sequence.others.classification_gnn import ClassificationGNN

def is_init(vectorizer, decl_vector):
    terms = vectorizer.devectorize(decl_vector, indices_only=True)
    return len(terms) == 1 and 'init' in terms[0] and terms[0]['init'] == 1

def load_data(data_dir, file_name, restrict = None):
    full_path = os.path.join(data_dir, file_name)
    with open("processed-data/output_vect.pkl", 'rb') as f:
        vectorizer = pickle.load(f)

    print("Loading data from %s" % full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)

    new_data = []
    num_inits = 0
    for d in data:
        g = {}
        g["graph"] = d["graph"]
        (data, indices, indptr, shape) = d["node_features"]
        g["node_features"] = csr_matrix((data, indices, indptr), shape)

        init = int(is_init(vectorizer, d['output_features']))
        num_inits += init
        g["output"] = np.eye(2)[init]
        new_data.append(g)
    
    cropped_data = []
    for graph_data in new_data:
        if graph_data['output'][1] == 0:
            if num_inits <= 0:
                continue
            cropped_data.append(graph_data)
            num_inits -= 0.75
        else:
            cropped_data.append(graph_data)

    return cropped_data


def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir') or './'
    train_data = args.get('--training-data') or "processed-data/graphs-body-decl-train.json"
    valid_data = args.get('--validation-data') or "processed-data/graphs-body-decl-valid.json"

    train_data = load_data(data_dir, train_data)
    valid_data = load_data(data_dir, valid_data)
    
    try:
        model = ClassificationGNN(args)
        model.train(train_data, valid_data)
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    main()
