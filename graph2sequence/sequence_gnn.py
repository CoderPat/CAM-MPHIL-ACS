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

from tensorflow.contrib.learn import ModeKeys
from scipy.sparse import vstack, csr_matrix
from .base.base_embeddings_gnn import BaseEmbeddingsGNN
from .base.utils import glorot_init, MLP, compute_bleu, compute_f1

class SequenceGNN(BaseEmbeddingsGNN):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.max_output_len = 0
        self.target_vocab_size = 0

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 50000,
            'decoder_rnn_cell': 'gru',         # (num_classes)
            'max_output_len': 10,
            'learning_rate': 0.005,

            'decoder_layers' : 1,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'num_timesteps': 4,

            'tie_fwd_bkwd': True,
            'attention': 'Luong'
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        self.placeholders['decoder_inputs'] = tf.placeholder(tf.int32, [None, self.max_output_len+1], name='decoder_inputs')
        self.placeholders['sequence_lens'] = tf.placeholder(tf.int32, [None], name='sequence_lens')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None, self.max_output_len], name='target_mask')
        self.placeholders['node_offsets'] = tf.placeholder(tf.int32, [None], name='node_offsets')
        super().prepare_specific_graph_model()


    def process_data(self, data, mode):
        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        if mode == ModeKeys.TRAIN and (self.params['input_shape'] is None or self.params['target_vocab_size'] is None):
            num_fwd_edge_types = 0
            for g in data:
                self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
                self.max_output_len = max(self.max_output_len, g['output'].shape[0] + 1)
                self.target_vocab_size = max(self.target_vocab_size, np.max(g['output'])+1 if len(g['output'])>0 else 0) 
                num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))


            self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
            self.annotation_size = max(self.annotation_size, data[0]["node_features"].shape[1])
            
            self.output_shape = (self.max_output_len, )
            self.go_symbol  = self.target_vocab_size
            self.eos_symbol = self.target_vocab_size + 1
            self.pad_symbol = self.target_vocab_size + 2

        return self.process_raw_graphs(data, mode)

    def process_raw_graphs(self, raw_data: Sequence[Any], mode) -> Any:
        processed_graphs = []
        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self._BaseEmbeddingsGNN__graph_to_adjacency_lists(d['graph'])
            processed_graphs.append({"adjacency_lists": adjacency_lists,
                                        "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                        "init": d["node_features"]})

            if mode != ModeKeys.INFER:
                sequence_len = d['output'].shape[0] + 1
                pad_size = self.max_output_len - sequence_len

                decoder_input = np.concatenate((np.array([self.go_symbol]), 
                                                d['output'],
                                                np.array([self.eos_symbol]),
                                                np.array(pad_size * [self.pad_symbol])))

                processed_graphs[-1].update({"targets": decoder_input[1:],
                                             "raw_targets": d['output'],
                                             "decoder_inputs": decoder_input,
                                             "sequence_len": sequence_len})

        return processed_graphs

    def make_minibatch_iterator(self, data: Any, mode):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if mode == ModeKeys.TRAIN:
            np.random.shuffle(data)
        elif mode == ModeKeys.EVAL:
            self.raw_targets = [graph['raw_targets'] for graph in data]

        # Pack until we cannot fit more graphs in the batch
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if mode == ModeKeys.TRAIN else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = csr_matrix(np.empty(shape=(0,data[0]['init'].shape[1])))
            batch_target_task_values = []
            batch_node_offsets = []
            batch_decoder_inputs = []
            batch_sequence_lens = []
            batch_target_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0
            
            while num_graphs < len(data) and node_offset + data[num_graphs]['init'].shape[0] < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = cur_graph['init'].shape[0]
                batch_node_features = vstack([batch_node_features, cur_graph['init']])
                batch_graph_nodes_list.extend((num_graphs_in_batch, i) for i in range(num_nodes_in_graph))
                batch_node_offsets.extend([node_offset]*num_nodes_in_graph)

                for i in range(self.num_edge_types):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                if mode != ModeKeys.INFER:
                    batch_target_task_values.append(cur_graph['targets'])
                    batch_decoder_inputs.append(cur_graph['decoder_inputs'])
                    batch_sequence_lens.append(cur_graph['sequence_len'])
                    batch_target_mask.append(cur_graph['sequence_len'] * [1] + (self.max_output_len - cur_graph['sequence_len'])*[0])

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_node_features = batch_node_features.tocoo()
            indices = np.array([[row] for row in batch_node_features.row])

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: (indices, batch_node_features.col, (batch_node_features.shape[0], )),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                self.placeholders['graph_nodes_list']: np.array(batch_graph_nodes_list, dtype=np.int32),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['node_offsets'] : np.array(batch_node_offsets)
            }

            if mode != ModeKeys.INFER:
                batch_feed_dict.update({
                    self.placeholders['target_values']: np.array(batch_target_task_values),
                    self.placeholders['decoder_inputs']: np.array(batch_decoder_inputs),
                    self.placeholders['sequence_lens']: np.array(batch_sequence_lens),
                    self.placeholders['target_mask'] : np.array(batch_target_mask),
                    self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                })


            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict


    def build_decoder_cell(self, last_h):
        h_dim = self.params['hidden_size']
        num_nodes = tf.shape(last_h, out_type=tf.int64)[0]
        num_graphs = tf.cast(self.placeholders['num_graphs'], tf.int64)

        padded_nodes = (self.placeholders['graph_nodes_list'][:, 1] + 
                        tf.cast(self.placeholders['node_offsets'], tf.int64))

        # from full node embeddings average embeddings per graph
        padded_node_list = tf.concat([tf.reshape(self.placeholders['graph_nodes_list'][:, 0], (-1, 1)),
                                      tf.reshape(padded_nodes, (-1, 1))], 
                                      axis=1)

        graph_mask = tf.SparseTensor(indices=padded_node_list,
                                      values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0], dtype=tf.float32),
                                      dense_shape=[num_graphs, num_nodes])

        graph_averages = (tf.sparse_tensor_dense_matmul(tf.cast(graph_mask, tf.float32), last_h)
                         / tf.reshape(tf.sparse_reduce_sum(graph_mask, 1), (-1, 1)))


        # and node embeddings for each graph
        padded_embeddings = tf.concat([tf.zeros([1, h_dim]), last_h], axis=0) 
        graph_nodes = tf.sparse_to_dense(self.placeholders['graph_nodes_list'],
                                        [num_graphs, self.max_num_vertices],
                                        padded_nodes)
        graph_embeddings = tf.nn.embedding_lookup(padded_embeddings, graph_nodes)
        
        #Build decoder cell
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

        if self.params['attention'] == 'Luong':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(h_dim, graph_embeddings,
                                                                    memory_sequence_length=None)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=h_dim)
            initial_state = decoder_cell.zero_state(self.placeholders['num_graphs'], tf.float32) \
                                        .clone(cell_state=graph_averages) 
            return decoder_cell, initial_state
        else:
            return decoder_cell, graph_averages


    def build_output(self, last_h):
        h_dim = self.params['hidden_size']

        self.projection_layer = tf.layers.Dense(self.target_vocab_size+3)

        self.weights['output_embeddings'] = tf.Variable(glorot_init([self.target_vocab_size+3, h_dim]),
                                                                     name='output_embeddings')

        decoder_cell, initial_state = self.build_decoder_cell(last_h)

        decoder_emb_inp = tf.nn.embedding_lookup(self.weights['output_embeddings'], 
                                                 self.placeholders['decoder_inputs'])

        with tf.variable_scope("train_decoder"):
            train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, 
                                                            self.placeholders['sequence_lens'], 
                                                            time_major=False)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                            train_helper, initial_state,
                                                            output_layer=self.projection_layer)

            train_outputs, _, _= tf.contrib.seq2seq.dynamic_decode(train_decoder)
        
        with tf.variable_scope("infer_decoder"):
            start_tokens = tf.fill([self.placeholders['num_graphs']], tf.cast(self.go_symbol, tf.int32))
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.weights['output_embeddings'], 
                                                                    start_tokens, self.eos_symbol)
   
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                            infer_helper, initial_state,
                                                            output_layer=self.projection_layer)

            self.ops['infer_output'], _, _= tf.contrib.seq2seq.dynamic_decode(infer_decoder,
                                                                              maximum_iterations=2*self.max_output_len)


        return train_outputs.rnn_output

    def get_loss(self, computed_logits, expected_outputs):
        batch_max_len = tf.reduce_max(self.placeholders['sequence_lens'])
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(expected_outputs[:, :batch_max_len], tf.int32),
                                                                  logits=computed_logits)

        return (tf.reduce_sum(crossent * self.placeholders['target_mask'][:, :batch_max_len]) / 
                tf.cast(self.placeholders['num_graphs'], tf.float32))


    def get_validation_ops(self, computed_logits, _ ):
        return [self.ops['infer_output'].sample_id]

    def get_inference_ops(self, computed_logits):
        return [self.ops['infer_output'].sample_id]

    def get_log(self, loss, speed, extra_results, mode):
        if mode == ModeKeys.TRAIN:
            return "loss : %.5f | instances/sec: %.2f", (loss, speed)

        elif mode == ModeKeys.EVAL:
            #Extract sampled sentences from results
            sampled_ids = [sampled_sentence.tolist() for result in extra_results for sampled_sentence in result[0]]

            #Calculate BLEU and F1
            reference_corpus = [reference.tolist() for reference in self.raw_targets]
            sampled_sentences = self.get_translations(sampled_ids)
            bleu = compute_bleu(reference_corpus, sampled_sentences, max_order=1)
            f1 = compute_f1(reference_corpus, sampled_sentences, unk_token=0)

            return "loss: %.5f | BLEU: %.5f | F1: %.5f | instances/sec: %.2f", (loss, bleu, f1, speed)

    def get_checkpoint_metric(self, valid_log):
        return valid_log[1]
    
    def get_translations(self, sampled_ids):
        sentences = []
        for sampled_sentence in sampled_ids:
            if self.eos_symbol in sampled_sentence:
                sampled_sentence = sampled_sentence[:sampled_sentence.index(self.eos_symbol)]
            sentences.append(sampled_sentence)
        return sentences

    def process_inference(self, infer_results, infer_graphs):
        sampled_ids = [sampled_sentence.tolist() for result in infer_results for sampled_sentence in result[0]]
        return self.get_translations(sampled_ids)


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


