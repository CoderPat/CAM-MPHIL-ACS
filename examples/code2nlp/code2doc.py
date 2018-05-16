"""
Usage:
    code2doc.py [options]

Options:
    -h --help                       Show this screen.
    --log_dir DIR                   Log dir name.
    --data_dir DIR                  Data dir name.
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

import numpy as np

import os
import json
import sys, traceback
import pdb
import pickle
import random

from graph2sequence.graph2seq import Graph2Seq

MAX_VERTICES_GRAPH = 700
MAX_OUTPUT_LEN = 100

CONFIG = {
    'batch_size': 25000,
    'graph_state_dropout_keep_prob': 0.8,
    'decoder_cells_dropout_keep_prob': 0.9,    
    'learning_rate': 0.001,
    'layer_timesteps': [2, 2, 2, 2],
    'attention_scope': None,
    'random_seed' : None,
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

        if len(new_data) >= 20000:
            break

        new_data.append(g)

    return new_data

def save_attention_map(coefs, src_labels, tgt_labels, path):
    coefs = coefs[:len(src_labels), :len(tgt_labels)]
    print(coefs)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(coefs*255, cmap=plt.cm.gray, alpha=0.8)
    ax.set_yticks(np.arange(coefs.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(coefs.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(tgt_labels, minor=False)
    ax.set_yticklabels(src_labels, minor=False)
    plt.xticks(rotation=90)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.grid(False)
    plt.savefig(path)


def main():
    args = docopt(__doc__)
    data_dir = args.get('--data-dir') or 'dataset/processed_data/'
    train_data = args.get('--training-data') or "graphs-func-doc-train.json"
    valid_data = args.get('--validation-data') or "graphs-func-doc-valid.json"
    test_data = args.get('--testing-data') or "graphs-func-doc-test.json"
    vects = args.get('--save-alignments')

    train_data = load_data(data_dir, train_data)
    valid_data = load_data(data_dir, valid_data)
    test_data = load_data(data_dir, test_data)

    if args.get('--no-train'):
        CONFIG['num_epochs'] = 0

    args['--config'] = json.dumps(CONFIG)

    try:
        model = Graph2Seq(args)
        model.train(train_data, valid_data)
        model.evaluate(test_data)

        if vects is not None:
            input_vect, output_vect = vects.split(',') 

            print("Loading output vectorizer")
            with open(output_vect, 'rb') as f:
                output_vect = pickle.load(f)

            print("Loading input vectorizer")
            with open(input_vect, 'rb') as f:
                input_vect = pickle.load(f)

            outputs, coefs = model.infer(test_data)
            outputs = output_vect.devectorize(outputs, subtokens=False)
            references = output_vect.devectorize([g['output'] for g in test_data], subtokens=False)

            for i, (output, coef, reference) in enumerate(zip(output, coefs, references)):
                save_attention_map(coef, reference, output, "attention_maps/attention_map_%d" % i)

    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    main()
