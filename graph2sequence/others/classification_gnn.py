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

from ..base.base_embeddings_gnn import BaseEmbeddingsGNN
from ..base.utils import glorot_init, MLP


class ClassificationGNN(BaseEmbeddingsGNN):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.accuracy_op = None

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'output_shape': None         # (num_classes)
        })
        return params

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gated_outputs = tf.nn.sigmoid(regression_gate(last_h)) * regression_transform(last_h)  # [v x c]

        # Sum up all nodes per-graph
        num_nodes = tf.shape(last_h, out_type=tf.int64)[0]
        graph_nodes = tf.SparseTensor(indices=self.placeholders['graph_nodes_list'],
                                      values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0], dtype=tf.float32),
                                      dense_shape=[self.placeholders['num_graphs'] , num_nodes])  # [g x v]
        return tf.sparse_tensor_dense_matmul(graph_nodes, gated_outputs) # [g c]


    def get_output(self, last_h):
        with tf.variable_scope("regression_gate"):
            self.weights['regression_gate'] = MLP(self.params['hidden_size'], self.output_shape[0], [],
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

    def get_extra_train_ops(self, computed_logits, target_classes):
        if self.accuracy is None:
            correct_prediction = tf.equal(tf.argmax(computed_logits,1), tf.argmax(target_classes,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return [self.accuracy]

    def get_extra_valid_ops(self, computed_logits, target_classes):
        if self.accuracy is None:
            correct_prediction = tf.equal(tf.argmax(computed_logits,1), tf.argmax(target_classes,1))
            self.accuracy = [tf.reduce_mean(tf.cast(correct_prediction, tf.float32))]
        return [self.accuracy]

    def get_accuracy(self, extra_results): 
        results, num_graphs = zip(*extra_results)
        accuracies = np.array([result[0] for result in results]) * np.array(num_graphs)
        return np.sum(accuracies)/np.sum(np.array(num_graphs))

    def get_train_log(self, loss, speed, extra_results):
        accuracy = self.get_accuracy(extra_results)
        return "Train: loss: %.5f | acc: %.5f | instances/sec: %.2f", (loss, accuracy, speed)
    
    def get_valid_log(self, loss, speed, extra_results):
        accuracy = self.get_accuracy(extra_results)
        return "Valid: loss: %.5f | acc: %.5f | instances/sec: %.2f", (loss, accuracy, speed)
        

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