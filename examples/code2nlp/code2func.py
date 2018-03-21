"""
Usage:
    code2func.py [options]

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
    --print-example FILE     Print random examples using the a vocabulary file
"""

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle

import numpy as np

import random
import os
import json
import sys, traceback
import pdb

from graph2sequence.sequence_gnn import Graph2SequenceGNN

def load_data(data_dir, file_name, restrict = None):
    full_path = os.path.join(data_dir, file_name)

    print("Loading data from %s" % full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)

    new_data = []
    for d in data:
        g = {}
        g["graph"] = d["graph"]
        (data, indices, indptr, shape) = d["node_features"]
        g["node_features"] = csr_matrix((data, indices, indptr), shape)
        g["output"] = np.array(d["output_features"])
        new_data.append(g)

    return new_data


def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir') or './'
    train_data = args.get('--training-data') or "processed-data/graphs-body-decl-train.json"
    valid_data = args.get('--validation-data') or "processed-data/graphs-body-decl-valid.json"
    output_vect = args.get('--print-example')

    train_data = load_data(data_dir, train_data)
    valid_data = load_data(data_dir, valid_data)


    try:
        model = Graph2SequenceGNN(args)
        model.train(train_data, valid_data)
        
        if output_vect is not None:
            print("Loading output vectorizer")
            with open(output_vect, 'rb') as f:
                output_vect = pickle.load(f)
            
            predicted_names = output_vect.devectorize(model.infer(valid_data), indices_only=True)
            true_names = output_vect.devectorize([g['output'] for g in valid_data], indices_only=True)
            
            print("printing random examples, press 'q' \to exit")
            while True:
                random_example = random.randint(0, len(predicted_names)-1)
                print("Predicted function name: %s" % "_".join(predicted_names[random_example]))
                print("True function name: %s" % "_".join(true_names[random_example]))
                if (input() == 'q'):
                    break
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    main()
