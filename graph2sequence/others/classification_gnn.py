#!/usr/bin/env/python
"""
Usage:
    classification_gnn.py [options]

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
import tensorflow as tf
import sys, traceback
import pdb

from .base.base_sparse_gnn import BaseSparseGNN
from .base.utils import glorot_init, MLP


class ClassificationGNN(BaseSparseGNN):
    """
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'output_shape': None         # (num_classes)
        })
        return params

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x c]

        # Sum up all nodes per-graph
        num_nodes = tf.shape(gate_input, out_type=tf.int64)[0]
        graph_nodes = tf.SparseTensor(indices=self.placeholders['graph_nodes_list'],
                                      values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0], dtype=tf.float32),
                                      dense_shape=[self.placeholders['num_graphs'] , num_nodes])  # [g x v]
        return tf.sparse_tensor_dense_matmul(graph_nodes, gated_outputs) # [g c]



    def get_output(self, last_h):
        with tf.variable_scope("regression_gate"):
            self.weights['regression_gate'] = MLP(2 * self.params['hidden_size'], self.output_shape[0], [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
        with tf.variable_scope("regression"):
            self.weights['regression_transform'] = MLP(self.params['hidden_size'], self.output_shape[0], [],
                                                                        self.placeholders['out_layer_dropout_keep_prob'])

        return self.gated_regression(self.ops['final_node_representations'],
                                     self.weights['regression_gate'],
                                     self.weights['regression_transform'])

    def get_loss(self, computed_logits, target_classes):
        cross_entropies = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=target_classes,
                                                                        logits=computed_logits))
        return tf.reduce_mean(cross_entropies)

    def get_accuracy(self, computed_logits, target_classes):
        correct_prediction = tf.equal(tf.argmax(computed_logits,1), tf.argmax(target_classes,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        


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