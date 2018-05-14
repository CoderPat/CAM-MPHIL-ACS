"""
Usage:
    code2doc_baseline.py [options]

Options:
    -h --help                Show this screen.
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --no-train               Sets epochs to zero (only for already trained models)
    --freeze-graph-model     Freeze weights of graph model components.
    --training-data FILE     Location of the training data
    --validation-data FILE   Location of the training data
    --restrict_data <value>
    --print-example FILE     Print random examples using the a vocabulary file

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
import pickle
import random

from graph2sequence.baselines.seq2seq import Seq2Seq

MAX_VERTICES_GRAPH = 1000
MAX_OUTPUT_LEN = 20

CONFIG = {
    'batch_size': 10000,
    'graph_state_dropout_keep_prob': 0.8,
    'decoder_cells_dropout_keep_prob': 0.9,    
    'learning_rate': 0.002,
    'attention_scope': None,
}

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
        g["node_types"] = np.array(d["node_types"])
        g["node_features"] = csr_matrix((data, indices, indptr), shape)
        g["output"] = np.array(d["output_features"])

        if shape[0] >= MAX_VERTICES_GRAPH or len(g["output"]) >= MAX_OUTPUT_LEN:
            continue

        new_data.append(g)

    return new_data


def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir') or 'dataset/processed_data/'
    train_data = args.get('--training-data') or "graphs-func-doc-train.json"
    valid_data = args.get('--validation-data') or "graphs-func-doc-valid.json"
    test_data = args.get('--testing-data') or "graphs-func-doc-test.json"
    input_vect = args.get('--print-alignment')
    output_vect = args.get('--print-example')

    train_data = load_data(data_dir, train_data)
    valid_data = load_data(data_dir, valid_data)

    if args.get('--no-train'):
        CONFIG['num_epochs'] = 0

    args['--config'] = json.dumps(CONFIG)

    try:
        model = Seq2Seq(args)
        model.train(train_data, valid_data)
        model.evaluate(test_data)

        if output_vect is not None:
            print("Loading output vectorizer")
            with open(output_vect, 'rb') as f:
                output_vect = pickle.load(f)

            predicted_names = output_vect.devectorize(model.infer(valid_data), indices_only=True)
            true_names = output_vect.devectorize([g['output'] for g in valid_data], indices_only=True)

            print("printing random examples, press 'q' \to exit")
            while True:
                random_example = random.randint(0, len(predicted_names)-1)
                print("Predicted function doc: %s" % " ".join(predicted_names[random_example]))
                print("True function doc: %s" % " ".join(true_names[random_example]))
                if (input() == 'q'):
                    break
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    main()
