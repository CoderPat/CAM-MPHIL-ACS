#!/usr/bin/env/python
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
import tensorflow as tf
import sys, traceback
import pdb

from scipy.sparse import vstack
from .base_gnn import BaseGNN
from .utils import glorot_init, SMALL_NUMBER


class BaseEmbeddingsGNN(BaseGNN):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 100000,
            'use_edge_bias': False,
            'use_edge_msg_avg_aggregation': True,

            'layer_timesteps': [1, 1, 1, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
        })
        return params

    def get_rnn_cell(self, cell_type, h_dim, activation_name, num_layers=1, 
                           dropout_keep_prob=1,
                           use_cudnn=False):
                           
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu

        if use_cudnn and activation_name == 'tanh':
            if cell_type == 'gru':
                decoder_cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, h_dim)
            elif cell_type == 'lstm':
                decoder_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, h_dim)
            
        else: 
            if cell_type == 'gru':
                create_raw_cell = lambda: tf.contrib.rnn.GRUCell(h_dim, activation=activation_fun)
            elif cell_type == 'lstm' and activation_name == 'tanh':
                create_raw_cell = lambda: tf.nn.rnn_cell.LSTMCell(h_dim, activation=activation_fun)

            create_cell = lambda: tf.nn.rnn_cell.DropoutWrapper(create_raw_cell(),
                                                                input_keep_prob=dropout_keep_prob,
                                                                state_keep_prob=dropout_keep_prob)

            if num_layers > 1:
                cells = [create_cell() for _ in range(num_layers)]
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            else:
                decoder_cell = create_cell()

        return decoder_cell


    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']

        self.placeholders['initial_node_representation'] = tf.sparse_placeholder(tf.int32, [None],
                                                                                 name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int64, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int64, [None, 2], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        cell_type = self.params['graph_rnn_cell'].lower()

        self.weights['input_embeddings'] = tf.Variable(glorot_init([self.annotation_size, h_dim]),
                                                                    name='input_embeddings')

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.weights['edge_weights'] = []
        self.weights['edge_biases'] = []
        self.weights['rnn_cells'] = []
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                self.weights['edge_weights'].append(tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                                                name='gnn_edge_weights_%i' % layer_idx))

                if self.params['use_edge_bias']:
                    self.weights['edge_biases'].append(tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                                                   name='gnn_edge_biases_%i' % layer_idx))

                
                encoder_cell = self.get_rnn_cell(cell_type, h_dim, activation_name, 1)
                self.weights['rnn_cells'].append(encoder_cell)

    def compute_final_node_representations(self) -> tf.Tensor:
        h_dim = self.params['hidden_size']
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int64)[0]

        cur_node_states = self.placeholders['initial_node_representation']  # number of nodes in batch v x V

        cur_node_states = tf.nn.embedding_lookup_sparse(self.weights['input_embeddings'], cur_node_states, None, combiner='mean')
        
        pad_len = ( tf.shape(self.placeholders['initial_node_representation'])[0] 
                  - tf.shape(cur_node_states)[0])
        cur_node_states = tf.concat([cur_node_states, tf.zeros((pad_len, h_dim))], axis=0)  


        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                adjacency_matrices = []  # type: List[tf.SparseTensor]
                for adjacency_list_for_edge_type in self.placeholders['adjacency_lists']:
                    # adjacency_list_for_edge_type (shape [-1, 2]) includes all edges of type e_type of a sparse graph with v nodes (ids from 0 to v).
                    adjacency_matrix_for_edge_type = tf.SparseTensor(indices=adjacency_list_for_edge_type,
                                                                     values=tf.ones_like(
                                                                         adjacency_list_for_edge_type[:, 1],
                                                                         dtype=tf.float32),
                                                                     dense_shape=[num_nodes, num_nodes])
                    adjacency_matrices.append(adjacency_matrix_for_edge_type)

                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        incoming_messages = []  # list of v x D

                        # Collect incoming messages per edge type
                        for adjacency_matrix in adjacency_matrices:
                            incoming_messages_per_type = tf.sparse_tensor_dense_matmul(adjacency_matrix,
                                                                                       cur_node_states)  # v x D
                            incoming_messages.extend([incoming_messages_per_type])

                        # Pass incoming messages through linear layer:
                        incoming_messages = tf.concat(incoming_messages, axis=1)  # v x [2 *] edge_types
                        messages_passed = tf.matmul(incoming_messages,
                                                    self.weights['edge_weights'][layer_idx])  # v x D

                        if self.params['use_edge_bias']:
                            messages_passed += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                         self.weights['edge_biases'][layer_idx])  # v x D

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)  # v x 1
                            messages_passed /= num_incoming_edges + SMALL_NUMBER

                        # pass updated vertex features into RNN cell
                        cur_node_states = self.weights['rnn_cells'][layer_idx](messages_passed, cur_node_states)[1]  # v x D

        return cur_node_states


    # ----- Data preprocessing and chunking into minibatches:

    def process_data(self, data, is_training_data: bool):
        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        if self.params['input_shape'] is None or self.params['output_shape'] is None:
            num_fwd_edge_types = 0
            for g in data:
                self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
                num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
            self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
            self.annotation_size = max(self.annotation_size, data[0]["node_features"].shape[1])
            self.output_shape = np.array(data[0]['output']).shape

        return self.process_raw_graphs(data, is_training_data)

    def process_raw_graphs(self, raw_data: Sequence[Any], mode) -> Any:
        processed_graphs = []
        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                     "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                     "init": d["node_features"]})
            if mode != ModeKeys.INFER:
                processed_graphs[-1]["labels"] = d["output"]

        return processed_graphs

    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0
            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = self.num_edge_types + edge_type
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    def make_minibatch_iterator(self, data: Any, mode):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if mode == ModeKeys.TRAIN:
            np.random.shuffle(data)

        # Pack until we cannot fit more graphs in the batch
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = None
            batch_target_task_values = []
            batch_target_task_mask = []
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

                target_task_values = []

                if not mode == ModeKeys.INFER:
                    batch_target_task_values.append(cur_graph['labels'])

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_node_features = batch_node_features.tocoo()
            indices = np.array([[row] for row in batch_node_features.row])

            if mode == ModeKeys.INFER:
                batch_feed_dict = {
                    self.placeholders['initial_node_representation']: (indices, batch_node_features.col, (batch_node_features.shape[0], )),
                    self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                    self.placeholders['graph_nodes_list']: np.array(batch_graph_nodes_list, dtype=np.int32),
                    self.placeholders['target_values']: np.array(batch_target_task_values),
                    self.placeholders['num_graphs']: num_graphs_in_batch,
                    self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                }
            else:
                batch_feed_dict = {
                    self.placeholders['initial_node_representation']: (indices, batch_node_features.col, (batch_node_features.shape[0], )),
                    self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                    self.placeholders['graph_nodes_list']: np.array(batch_graph_nodes_list, dtype=np.int32),
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
