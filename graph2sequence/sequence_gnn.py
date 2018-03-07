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

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'decoder_rnn_cell': 'lstm'         # (num_classes)
        })
        return params

    def get_output(self, last_h):
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

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, 
                                                   decoder_lengths, time_major=False)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                  helper, encoder_state, output_layer=projection_layer)
        # Dynamic decoding
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        logits = outputs.rnn_output

    def get_loss(self, computed_logits, target_classes):

    def get_accuracy(self, computed_logits, target_classes):

        


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