"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
"""

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict
from scipy.sparse import csr_matrix

import numpy as np

import os
import json
import sys, traceback
import pdb

from graph2sequence.others.classification_gnn import ClassificationGNN

def is_init(vectorizer, decl_vector):
    terms = vectorizer.devectorize(decl_vector, indices_only=True)
    return len(terms) == 1 and 'init' in terms and terms['init'] == 1

def load_data(data_dir, file_name, restrict = None):
    full_path = os.path.join(data_dir, file_name)
    if save_output_vect is not None:
        with open("processed-data/output-vect.pkl", 'rb') as f:
            vectorizer = pickle.load(f)

    print("Loading data from %s" % full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)

    new_data = []
    for d in data:
        g = {}
        g["graph"] = d["graph"]
        (data, indices, indptr, shape) = d["node_features"]
        g["node_features"] = csr_matrix((data, indices, indptr), shape)
        g["output"] = np.eye(2)[int(is_init(vectorizer, d['output_features']))]
        new_data.append(g)

    return new_data


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
