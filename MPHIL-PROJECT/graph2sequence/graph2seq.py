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
from collections import defaultdict, Counter
import numpy as np
import tensorflow as tf
import sys
import traceback
import pdb

from tensorflow.contrib.learn import ModeKeys
from scipy.sparse import vstack, csr_matrix
from .base.base_embeddings_gnn import BaseEmbeddingsGNN
from .base.utils import glorot_init, MLP, compute_bleu, compute_f1, compute_acc

class Graph2Seq(BaseEmbeddingsGNN):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.max_output_len = 0
        self.target_vocab_size = 0
        self.writer = None
        self.max_scoped_vertices = 0

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'learning_rate': 0.001,
            'num_timesteps': 4,
            'hidden_size': 256,
            'tie_fwd_bkwd': False,
            'use_edge_bias': True,

            'decoder_layers' : 1,
            'decoder_rnn_cell': 'LSTM',                  # (num_classes)
            'decoder_num_units': 512,                   # doesn't work yet
            'decoder_rnn_activation': 'tanh',
            'decoder_cells_dropout_keep_prob': 0.9,
            'beam_width': 1,

            'attention': 'Luong',
            'attention_scope': None,

            'bleu': [1, 4],
            'f1': True,
            'acc': True,
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        self.placeholders['graph_sizes'] = tf.placeholder(tf.int32, [None], name='graph_sizes')
        self.placeholders['scoped_size'] = tf.placeholder(tf.int32, [None], name='scoped_indexes')        
        self.placeholders['scoped_indexes'] = tf.placeholder(tf.int32, [None, self.max_scoped_vertices, 2], name='scoped_indexes')
        self.placeholders['decoder_inputs'] = tf.placeholder(tf.int32, [None, self.max_output_len+1], name='decoder_inputs')
        self.placeholders['decoder_keep_prob'] = tf.placeholder(tf.float32, None, name='dropout_keep_prob')
        self.placeholders['sequence_lens'] = tf.placeholder(tf.int32, [None], name='sequence_lens')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None, self.max_output_len], name='target_mask')
        super().prepare_specific_graph_model()


    def process_data(self, data, mode):
        restrict = self.args.get("--restrict_data")
        if restrict is not None and int(restrict) > 0:
            data = data[:int(restrict)]

        # Get some common data out:
        if mode == ModeKeys.TRAIN and (self.params['input_shape'] is None or self.params['target_vocab_size'] is None):
            num_fwd_edge_types = 0
            for g in data:
                self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]])+1)
                self.max_output_len = max(self.max_output_len, g['output'].shape[0] + 1)
                self.target_vocab_size = max(self.target_vocab_size, np.max(g['output'])+1 if len(g['output'])>0 else 0) 
                num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
                if self.params['attention_scope'] is not None:
                    if isinstance(self.params['attention_scope'], int):
                        self.max_scoped_vertices = max(self.max_scoped_vertices, 
                                                        Counter(g['node_types'])[self.params['attention_scope']])
                    elif isinstance(self.params['attention_scope'], List):
                        counter = Counter(g['node_types'])
                        self.max_scoped_vertices = max(self.max_scoped_vertices,
                                                       sum([counter[scope] for scope in self.params['attention_scope']]))
                    


                             
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

            if self.params['attention_scope'] is not None:
                if isinstance(self.params['attention_scope'], int):
                    scoped_indexes = np.where(d['node_types'] == self.params['attention_scope'])[0]
                elif isinstance(self.params['attention_scope'], List):
                    scoped_indexes = np.where(np.isin(d['node_types'], self.params['attention_scope']))[0]
                processed_graphs[-1]['scoped_indexes'] = scoped_indexes



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
        graph_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if mode == ModeKeys.TRAIN else 1.
        decoder_dropout_keep_prob = self.params['decoder_cells_dropout_keep_prob'] if mode == ModeKeys.TRAIN else 1.
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = csr_matrix(np.empty(shape=(0,data[0]['init'].shape[1])))
            batch_graph_sizes = []
            batch_scoped_indexes = []
            batch_scoped_len = []
            batch_target_task_values = []
            batch_decoder_inputs = []
            batch_sequence_lens = []
            batch_target_mask = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0
            
            final_graph_nodes = data[num_graphs]['init'].shape[0] if self.params['attention'] is None \
                                                                  else self.max_num_vertices
            while num_graphs < len(data) and node_offset + final_graph_nodes < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = cur_graph['init'].shape[0]
                batch_node_features = vstack([batch_node_features, cur_graph['init']])

                if self.params['attention'] is None:
                    batch_graph_nodes_list.extend((num_graphs_in_batch, node_offset + i) for i in range(num_nodes_in_graph))
                else:
                    pad_len = self.max_num_vertices - num_nodes_in_graph
                    pad_nodes = csr_matrix((pad_len, cur_graph['init'].shape[1]), dtype=np.int32)
                    batch_node_features = vstack([batch_node_features, pad_nodes])
                    batch_graph_sizes.append(num_nodes_in_graph)
                    num_nodes_in_graph = self.max_num_vertices 

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

                if self.params['attention_scope'] is not None:
                    scoped_len = cur_graph['scoped_indexes'].shape[0]
                    pad_size = self.max_scoped_vertices - scoped_len
                    padded_scoped = np.concatenate((cur_graph['scoped_indexes'] + 1, 
                                                   np.array(pad_size * [0])))
                    padded_scoped = np.concatenate([[[num_graphs_in_batch]]*self.max_scoped_vertices, 
                                                   padded_scoped[:,np.newaxis]], axis=1)
                    batch_scoped_indexes.append(padded_scoped)
                    batch_scoped_len.append(scoped_len)

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_node_features = batch_node_features.tocoo()
            indices = np.array([[row] for row in batch_node_features.row])

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: (indices, batch_node_features.col, (batch_node_features.shape[0], )),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: graph_dropout_keep_prob,
                self.placeholders['decoder_keep_prob']: decoder_dropout_keep_prob,
                self.placeholders['graph_sizes'] : np.array(batch_graph_sizes)
            }
            if self.params['attention'] is None:
                batch_feed_dict[self.placeholders['graph_nodes_list']] = np.array(batch_graph_nodes_list, dtype=np.int32)
            else:
                batch_feed_dict[self.placeholders['graph_sizes']] = np.array(batch_graph_sizes)
            
            if mode != ModeKeys.INFER:
                batch_feed_dict.update({
                    self.placeholders['target_values']: np.array(batch_target_task_values),
                    self.placeholders['decoder_inputs']: np.array(batch_decoder_inputs),
                    self.placeholders['sequence_lens']: np.array(batch_sequence_lens),
                    self.placeholders['target_mask'] : np.array(batch_target_mask),
                })

            if self.params['attention_scope'] is not None:
                batch_feed_dict[self.placeholders['scoped_indexes']] = batch_scoped_indexes
                batch_feed_dict[self.placeholders['scoped_size']] = batch_scoped_len


            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

    
    def extract_from_graph(self, last_h):
        if self.params['attention'] is None:
            #Extract directly from compressed node embeddings matrix
            graph_mask = tf.SparseTensor(
                indices=self.placeholders['graph_nodes_list'],
                values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0], dtype=tf.float32),
                dense_shape=[tf.cast(self.placeholders['num_graphs'], tf.int64) , tf.cast(tf.shape(last_h)[0], tf.int64)])  # [g x v]
                
            graph_averages = (tf.sparse_tensor_dense_matmul(tf.cast(graph_mask, tf.float32), last_h)
                              / tf.reshape(tf.sparse_reduce_sum(graph_mask, 1), (-1, 1)))

            return graph_averages, None

        else:
            #Reshape to a 3D tensor as [batch, nodes, h_dim]
            graph_embeddings = tf.reshape(last_h, shape=(self.placeholders['num_graphs'], 
                                                         self.max_num_vertices,
                                                         self.params['decoder_num_units'])) 

            graph_averages = (tf.reduce_sum(graph_embeddings, axis=1) / 
                              tf.reshape(tf.cast(self.placeholders["graph_sizes"], tf.float32), (-1, 1)))

            return graph_averages, graph_embeddings


    def build_decoder_cell(self, graph_averages, graph_embeddings, beam_tile=False):
        h_dim = self.params['decoder_num_units']

        activation_name = self.params['decoder_rnn_activation'].lower()
        cell_type = self.params['decoder_rnn_cell'].lower()
        num_layers = self.params['decoder_layers']
        dropout = self.params['decoder_cells_dropout_keep_prob']

        decoder_cell = self.get_rnn_cell(cell_type, h_dim, activation_name, num_layers, 
                                         self.placeholders['decoder_keep_prob'])

        # Generate initial state for decoder in case it's an LSTM
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(graph_averages, graph_averages) if cell_type == 'lstm' \
                        else graph_averages

        # Pad initial state in case of multilayer decoder
        if num_layers > 1:
            if cell_type == 'lstm':
                pad = [tf.nn.rnn_cell.LSTMStateTuple(tf.zeros((self.placeholders['num_graphs'],h_dim)),
                                                     tf.zeros((self.placeholders['num_graphs'],h_dim)))
                                            for _ in range(num_layers-1)]

                initial_state = tuple([initial_state] + pad)
            else:
                pad = [tf.zeros((self.placeholders['num_graphs'],h_dim)) for _ in range(num_layers-1)]
                initial_state = tuple([initial_state] + pad)

        
        #Wrap in attention
        if self.params['attention'] is not None:

            # Deal with attention scopping
            if self.params['attention_scope'] is not None:
                pad_embedding = tf.zeros((self.placeholders['num_graphs'], 1, h_dim), tf.float32)
                padded_embeddings = tf.concat([pad_embedding, graph_embeddings], axis=1)
                attention_memory = tf.gather_nd(padded_embeddings, self.placeholders['scoped_indexes'])
                memory_length = self.placeholders["scoped_size"]
            else:
                attention_memory = graph_embeddings
                memory_length = self.placeholders["graph_sizes"]

            if beam_tile:
                attention_memory = tf.contrib.seq2seq.tile_batch(attention_memory, self.params['beam_width'])
                graph_averages = tf.contrib.seq2seq.tile_batch(graph_averages, self.params['beam_width'])
                memory_length = tf.contrib.seq2seq.tile_batch(memory_length, self.params['beam_width'])

            # Pick attention mechanism
            if self.params['attention'] == 'Luong':
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    h_dim, 
                    attention_memory,
                    memory_sequence_length=memory_length,
                    scale=True)
            elif self.params['attention'] == 'Bahdanau':
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    h_dim, 
                    attention_memory,
                    memory_sequence_length=memory_length,
                    normalize=True)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, 
                attention_mechanism,
                attention_layer_size=h_dim,
                alignment_history=True)

    
            mult = self.params['beam_width'] if beam_tile else 1
            initial_state = decoder_cell.zero_state(self.placeholders['num_graphs'] * mult, tf.float32).clone(cell_state=initial_state)   

        return decoder_cell, initial_state

    def build_output(self, last_h):
        encoder_units = self.params['hidden_size']
        decoder_units = self.params['decoder_num_units']

        self.projection_layer = tf.layers.Dense(self.target_vocab_size+3)
        self.projection_layer.build(decoder_units)
        self.projection_weights = self.projection_layer.kernel

        self.weights['bridge'] = tf.Variable(glorot_init([encoder_units, decoder_units]),
                                                          name='bridge')

        bridged_h = tf.matmul(last_h, self.weights['bridge'])

        decoder_emb_inp = tf.nn.embedding_lookup(tf.transpose(self.projection_weights), 
                                                 self.placeholders['decoder_inputs'])

        graph_average, graph_embeddings = self.extract_from_graph(bridged_h)

        # train decoding 
        with tf.variable_scope("decoder"):

            decoder_cell, initial_state = self.build_decoder_cell(graph_average, graph_embeddings)

            train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=decoder_emb_inp,
                sampling_probability=0.1, 
                sequence_length=self.placeholders['sequence_lens'],
                embedding=tf.transpose(self.projection_weights),
                time_major=False)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, 
                helper=train_helper, 
                initial_state=initial_state,
                output_layer=self.projection_layer)

            train_outputs = tf.contrib.seq2seq.dynamic_decode(train_decoder)[0].rnn_output
        
        with tf.variable_scope("decoder", reuse=True):

            decoder_cell, initial_state = self.build_decoder_cell(graph_average, graph_embeddings, beam_tile=True)

            start_tokens = tf.fill([self.placeholders['num_graphs']], tf.cast(self.go_symbol, tf.int32))

            infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, 
                embedding=tf.transpose(self.projection_weights),
                start_tokens=start_tokens, 
                end_token=self.eos_symbol,
                initial_state=initial_state,
                beam_width=self.params['beam_width'],
                output_layer=self.projection_layer)

            decode_state = tf.contrib.seq2seq.dynamic_decode(infer_decoder,
                                                             maximum_iterations=2*self.max_output_len)
            self.ops['infer_output'], self.ops['final_context_state'], _ = decode_state

            self.ops['alignment_history'] = self.create_attention_coefs(self.ops['final_context_state']) if self.params['attention'] \
                                                    else tf.no_op()

        return train_outputs

    def create_attention_coefs(self, final_context_state):
        """create attention image and attention summary."""
        attention_images = (final_context_state.cell_state.alignment_history)
        # Reshape to (batch, src_seq_len, tgt_seq_len)
        attention_images = tf.transpose(attention_images, [1, 2, 0])
        return attention_images

    def get_loss(self, computed_logits, expected_outputs):
        decoder_units = self.params['decoder_num_units']
        batch_max_len = tf.reduce_max(self.placeholders['sequence_lens'])

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(expected_outputs[:, :batch_max_len], tf.int32),
                                                                    logits=computed_logits)


        return (tf.reduce_sum(crossent * self.placeholders['target_mask'][:, :batch_max_len]) / 
                               tf.cast(self.placeholders['num_graphs'], tf.float32))


    def get_validation_ops(self, computed_logits, _ ):
        return [tf.transpose(self.ops['infer_output'].predicted_ids, (0,2,1))]

    def get_inference_ops(self, computed_logits):
        return [tf.transpose(self.ops['infer_output'].predicted_ids, (0,2,1)), self.ops['alignment_history']]

    def get_log(self, loss, speed, extra_results, mode):
        if mode == ModeKeys.TRAIN:
            return "loss : %.5f | instances/sec: %.2f", (loss, speed)

        elif mode == ModeKeys.EVAL:
            format_string, metrics = "loss: %.5f", (loss,)

            #Extract sampled sentences from results
            sampled_ids = [sampled_sentence[0, :].tolist() for result in extra_results for sampled_sentence in result[0]]

            #Calculate BLEU and F1
            reference_corpus = [reference.tolist() for reference in self.raw_targets]
            sampled_sentences = self.get_translations(sampled_ids)
            for n in self.params['bleu']:
                bleu = compute_bleu(reference_corpus, sampled_sentences, max_order=n)
                format_string += (" | BLEU-%d: " % n) + "%.5f"
                metrics = metrics + (bleu, )

            if self.params['f1']:
                f1 = compute_f1(reference_corpus, sampled_sentences, unk_token=0)
                format_string += " | F1: " + "%.5f"
                metrics = metrics + (f1, )
            
            if self.params['acc']:
                acc = compute_acc(reference_corpus, sampled_sentences, unk_token=0)
                format_string += " | acc.: " + "%.5f"
                metrics = metrics + (acc, )

            format_string += " | instances/sec: %.2f"
            metrics = metrics + (speed, )

            return format_string, metrics

    def get_checkpoint_metric(self, valid_log):
        return valid_log[1]
    
    def get_translations(self, sampled_ids):
        sentences = []
        for sampled_sentence in sampled_ids:
            if self.eos_symbol in sampled_sentence:
                sampled_sentence = sampled_sentence[:sampled_sentence.index(self.eos_symbol)]
            if self.pad_symbol in sampled_sentence:
                sampled_sentence = sampled_sentence[:sampled_sentence.index(self.pad_symbol)]
            if self.go_symbol in sampled_sentence:
                sampled_sentence = sampled_sentence[:sampled_sentence.index(self.go_symbol)] 
            sentences.append(sampled_sentence)
        return sentences

    def infer(self, input_data, alignment_history=False):
        res = super().infer(input_data)
        if alignment_history:
            return res
        else:
            return res[0]


    def process_inference(self, infer_results, coefs=False):
        coefs = [coef for result in infer_results for coef in result[1]]
        sampled_ids = [sampled_sentence.tolist() for result in infer_results for sampled_sentence in result[0]]
        
        return [self.get_translations(ids) for ids in sampled_ids], coefs


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


