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

from scipy.sparse import vstack
from .base.base_embeddings_gnn import BaseEmbeddingsGNN
from .base.utils import glorot_init, MLP


class Graph2SequenceGNN(BaseEmbeddingsGNN):
    """
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'decoder_rnn_cell': 'gru'         # (num_classes)
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        self.placeholders['decoder_inputs'] = tf.placeholder(tf.int32, [None, self.max_output_len+1], name='decoder_inputs')
        self.placeholders['sequence_lens'] = tf.placeholder(tf.int32, [None, ], name='sequence_lens')
        self.placeholders['target_weights'] = tf.placeholder(tf.float32, [None, self.max_output_len], name='target_weights')
        super().prepare_specific_graph_model()


    def process_data(self, data, is_training_data: bool):
        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        if self.params['input_shape'] is None or self.params['target_vocab_size'] is None:
            num_fwd_edge_types = 0
            self.max_output_len = 0
            self.target_vocab_size = 0
            for g in data:
                self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
                self.max_output_len = max(self.max_output_len, g['output'].shape[0])
                self.target_vocab_size = max(self.target_vocab_size, np.max(data[0]['output'])+1) 
                num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))


            self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
            self.annotation_size = max(self.annotation_size, data[0]["node_features"].shape[1])
            
            self.output_shape = (self.max_output_len, )
            self.go_symbol = self.target_vocab_size
            self.pad_symbol = self.target_vocab_size + 1

        return self.process_raw_graphs(data, is_training_data)

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        processed_graphs = []
        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self._BaseEmbeddingsGNN__graph_to_adjacency_lists(d['graph'])
            sequence_len = d['output'].shape[0]
            pad_size = self.max_output_len - sequence_len
            decoder_input = np.concatenate((np.array([self.go_symbol]), 
                                            d['output'], 
                                            np.array(pad_size * [self.pad_symbol])))
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"],
                                     "targets": decoder_input[1:],
                                     "decoder_inputs": decoder_input,
                                     "sequence_len": sequence_len})

        return processed_graphs

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = None
            batch_target_task_values = []
            batch_decoder_inputs = []
            batch_target_task_mask = []
            batch_sequence_lens = []
            batch_target_weights = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0
            
            while num_graphs < len(data) and node_offset + data[num_graphs]['init'].shape[0] < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = cur_graph['init'].shape[0]
                batch_node_features = vstack([batch_node_features, cur_graph['init']]) if num_graphs >= 1 else cur_graph['init']
                batch_graph_nodes_list.extend(
                    (num_graphs_in_batch, node_offset + i) for i in range(num_nodes_in_graph))
                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                batch_target_task_values.append(cur_graph['targets'])
                batch_decoder_inputs.append(cur_graph['decoder_inputs'])
                batch_sequence_lens.append(cur_graph['sequence_len'])
                batch_target_weights.append(cur_graph['sequence_len'] * [1] + (self.max_output_len - cur_graph['sequence_len'])*[0])
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_node_features = batch_node_features.tocoo()
            indices = np.array([[row] for row in batch_node_features.row])
            batch_feed_dict = {
                self.placeholders['initial_node_representation']: (indices, batch_node_features.col, (batch_node_features.shape[0], )),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                self.placeholders['graph_nodes_list']: np.array(batch_graph_nodes_list, dtype=np.int32),
                self.placeholders['target_values']: np.array(batch_target_task_values),
                self.placeholders['decoder_inputs']: np.array(batch_decoder_inputs),
                self.placeholders['sequence_lens']: np.array(batch_sequence_lens),
                self.placeholders['target_weights'] : np.array(batch_target_weights),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

    def get_output(self, last_h):
        h_dim = self.params['hidden_size']
        projection_layer = tf.layers.Dense(self.target_vocab_size+2, use_bias=False)
        projection_layer.build([None, h_dim])
        output_embeddings = projection_layer.kernel

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
    
        cell_type = self.params['decoder_rnn_cell'].lower()
        if cell_type == 'gru':
            decoder_cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == 'lstm':
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            decoder_cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)

        num_nodes = tf.shape(last_h, out_type=tf.int64)[0]
        graph_nodes = tf.SparseTensor(indices=self.placeholders['graph_nodes_list'],
                                      values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0], dtype=tf.float32),
                                      dense_shape=[self.placeholders['num_graphs'] , num_nodes])  # [g x v]

        graph_embeddings = (tf.sparse_tensor_dense_matmul(tf.cast(graph_nodes, tf.float32), last_h)
                            / tf.reshape(tf.sparse_reduce_sum(graph_nodes, 1), (-1, 1)))

        decoder_emb_inp = tf.nn.embedding_lookup(tf.transpose(output_embeddings), 
                                                 self.placeholders['decoder_inputs'])

        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, 
                                                   self.placeholders['sequence_lens'], 
                                                   time_major=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                  helper, graph_embeddings,
                                                  output_layer=projection_layer)

        outputs, _, _= tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        return outputs.rnn_output

    def get_loss(self, computed_logits, expected_outputs):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(expected_outputs, tf.int32),
                                                                  logits=computed_logits)
        return tf.reduce_sum(crossent * self.placeholders['target_weights'] ) / tf.cast(self.placeholders['num_graphs'], tf.float32)


    def get_accuracy(self, computed_logits, expected_outputs):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(expected_outputs, tf.int32), 
                                                                  logits=computed_logits)
        return tf.exp(tf.reduce_sum(crossent * self.placeholders['target_weights'] ) / tf.cast(self.placeholders['num_graphs'], tf.float32))
 


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