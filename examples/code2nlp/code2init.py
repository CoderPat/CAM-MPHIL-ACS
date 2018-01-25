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

import numpy as np

import os
import json
import sys, traceback
import pdb

from graph2sequence.classification_gnn import ClassificationGNN

def load_data(data_dir, file_name, restrict = None):
    full_path = os.path.join(data_dir, file_name)

    print("Loading data from %s" % full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)

    new_data = []
    for d in data:
        g = {}
        g["graph"] = d["graph"]
        g["node_features"] = np.eye(51)[np.array(d["node_features"])]
        g["output"] = np.eye(2)[int(d["label"])]
        new_data.append(g)

    return new_data


def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir')
    train_data = args.get('--training-data')
    valid_data = args.get('--validation-data')
    if data_dir is None:
        data_dir = "./"
    if train_data is None:
        train_data = "processed-data/graphs-train.json"
    if valid_data is None:
        valid_data = "processed-data/graphs-valid.json"

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