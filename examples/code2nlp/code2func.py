"""
Usage:
    code2func.py [options]

Options:
    -h --help                Show this screen.
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --no-train               Sets epochs to zero (only for already trained models)
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

from graph2sequence.sequence_gnn import SequenceGNN

MAX_VERTICES_IN_GRAPH = 1000

CONFIG = {
    'batch_size': 5000,
    'graph_state_dropout_keep_prob': 0.7,
    'decoder_cells_dropout_keep_prob': 0.9,
}

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

        if shape[0] >= MAX_VERTICES_IN_GRAPH:
            continue 

        g["node_features"] = csr_matrix((data, indices, indptr), shape)
        g["output"] = np.array(d["output_features"])
        new_data.append(g)

    print("Using %d out of %d" % (len(new_data), len(raw_data)))
    return new_data


def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir') or './'
    train_data = args.get('--training-data') or "processed-data/graphs-body-decl-train.json"
    valid_data = args.get('--validation-data') or "processed-data/graphs-body-decl-valid.json"
    output_vect = args.get('--print-example')

    train_data = load_data(data_dir, train_data)
    valid_data = load_data(data_dir, valid_data)

    if args.get('--no-train'):
        CONFIG['num_epochs'] = 0
        
    args['--config'] =  json.dumps(CONFIG)

    try:
        model = SequenceGNN(args)
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
