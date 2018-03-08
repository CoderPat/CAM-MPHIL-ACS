#!/usr/bin/env/python
"""
Usage:
    sequence_gnn.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from
    --freeze-graph-model     Freeze weights of graph model components.
"""

from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb

from ..base.base_embeddings_gnn import BaseEmbeddingsGNN
from ..base.utils import glorot_init, MLP


class Graph2SequenceGNN(BaseEmbeddingsGNN):
    """
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'decoder_rnn_cell': 'lstm'         # (num_classes)
        })
        return params

    def process_data(self, data, is_training_data: bool):
        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        if self.params['input_shape'] is None or self.params['target_vocab_size'] is None:
            num_fwd_edge_types = 0
            for g in data:
                self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
                num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
            self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
            self.annotation_size = max(self.annotation_size, data[0]["node_features"].shape[1])
            self.target_vocab_size =  data[0]['output'].shape[1]

        return self.process_raw_graphs(data, is_training_data)

    def get_output(self, last_h):
        h_dim = self.params['hidden_size']

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
    
        cell_type = self.params['decoder_rnn_cell'].lower()
        if cell_type == 'gru':
            decoder_cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            decoder_cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)


        graph_embeddings = (tf.cast(ph1, tf.float32) @ ph2) / 
                            tf.reshape(tf.cast(tf.reduce_sum(ph1, 1), tf.float32), (-1, 1))
                                
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, 
                                                   decoder_lengths, time_major=False)
        projection_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                  helper, graph_embeddings
                                                  output_layer=projection_layer)

        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        logits = outputs.rnn_output

    def get_loss(self, computed_logits, expected_outputs):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=expected_outputs, 
                                                                  logits=computed_logits)
        return tf.reduce_sum(crossent * target_weights) / self.placeholders['num_graphs']


    def get_accuracy(self, computed_logits, target_classes):
        pass
 


def main():
    try:
        model = ClassificationGNN(args)
        model.train(train_data, valid_data)
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()