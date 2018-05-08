import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random

from tensorflow.contrib.learn import ModeKeys

from scipy.sparse import vstack, csr_matrix
from ..base.utils import ThreadedIterator, SMALL_NUMBER
from ..base.utils import glorot_init, MLP, compute_bleu, compute_f1

class Seq2Seq(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.0005,
            'clamp_gradient_norm': 5.0,

            'embedding_size': 512,
            'state_size': 512,

            'random_seed': 0,
            'max_output_len': 30,

            'input_shape': None,        # (max_num_vertices, num_edges_types, annotation_size)
            'output_shape': None        # for subclasses to decide
        } 

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))

        
        # Build the actual model
        if params['input_shape'] is not None and params['output_shape'] is not None:
            self.max_num_vertices, self.num_edge_types, self.annotation_size = params['input_shape']
            self.output_shape = params['output_shape']
            self.build_model()
        else:
            self.graph = None

    def build_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            if self.params['random_seed'] is not None:
                random.seed(self.params['random_seed'])
                np.random.seed(self.params['random_seed'])
                tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = self.args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        self.sess.run(init_op)

    def make_model(self):
        self.placeholders['encoder_inputs'] = tf.sparse_placeholder(
            tf.int32, None, 
            name="encoder_inputs")
        self.placeholders['encoder_lengths'] = tf.placeholder(
            tf.int32, [None], 
            name="encoder_lengths")
        self.placeholders['decoder_inputs'] = tf.placeholder(
            tf.int32, [None, self.max_output_len + 1], 
            name="decoder_inputs")
        self.placeholders['decoder_targets'] = tf.placeholder(
            tf.int32, [None, self.max_output_len], 
            name="decoder_targets")
        self.placeholders['decoder_lengths'] = tf.placeholder(
            tf.int32, [None], 
            name="decoder_lengths")
        self.placeholders['target_mask'] = tf.placeholder(
            tf.int32, [None, self.max_output_len ], 
            name="target_mask")

        self.batch_size = tf.shape(self.placeholders['encoder_lengths'])[0]
        self.build_encoder()
        self.build_decoders()

    def build_encoder(self):
        encoder_embeddings = tf.get_variable(
            "encoder_embeddings", [self.src_vocab_size, self.params['embedding_size']])
        
        embedded_encoder_inp = tf.nn.embedding_lookup_sparse(
            encoder_embeddings, self.placeholders['encoder_inputs'], 
            sp_weights=None, 
            combiner='mean')

        pad_len = (tf.shape(self.placeholders['encoder_inputs'])[0] 
                   - tf.shape(embedded_encoder_inp)[0])

        embedded_encoder_inp = tf.concat(
            [embedded_encoder_inp, tf.zeros((pad_len, self.params['embedding_size']))], axis=0)  

        embedded_encoder_inp = tf.reshape(embedded_encoder_inp, (self.batch_size,
                                                                 self.max_input_len,
                                                                 self.params['embedding_size']))

        fwd_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['state_size']/2)
        bwd_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['state_size']/2)

        output, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fwd_encoder_cell,
            cell_bw=bwd_encoder_cell,
            inputs=embedded_encoder_inp,
            sequence_length=self.placeholders['encoder_lengths'],
            dtype=tf.float32)
        
        (fwd_c, fwd_h), (bwd_c, bwd_h) = state
        self.encoder_state = tf.contrib.rnn.LSTMStateTuple(
            tf.concat([fwd_c, bwd_c], axis=1), tf.concat([fwd_c, bwd_c], axis=1))
        self.encoder_output = tf.concat(output, axis=2)

    def build_decoders(self):
        self.decoder_embeddings = tf.get_variable(
            "decoder_embeddings", [self.tgt_vocab_size+3, self.params['embedding_size']])

        self.projection_layer = tf.layers.Dense(
            self.tgt_vocab_size+3, use_bias=False)

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['state_size'])

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.params['state_size'], self.encoder_output,
            memory_sequence_length=self.placeholders['encoder_lengths'])

        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=self.params['state_size'])

        self.initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32).\
            clone(cell_state=self.encoder_state)
        
        self.build_train_decoder()
        self.build_infer_decoder()

    def build_train_decoder(self):
        embedded_decoder_inp = tf.nn.embedding_lookup(
            self.decoder_embeddings, self.placeholders['decoder_inputs'])

        helper = tf.contrib.seq2seq.TrainingHelper(
            embedded_decoder_inp, self.placeholders['decoder_lengths'])

        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, self.initial_state,
            output_layer=self.projection_layer)
        
        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        
        logits = final_outputs.rnn_output
        batch_max_len = tf.reduce_max(self.placeholders['decoder_lengths'])

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.placeholders['decoder_targets'][:, :batch_max_len], 
            logits=logits)

        target_mask = tf.cast(self.placeholders['target_mask'], tf.float32)[:, :batch_max_len]
        self.train_loss = (tf.reduce_sum(crossent  * target_mask) 
                           / tf.cast(self.batch_size, tf.float32))

    def build_infer_decoder(self):
        start_tokens =  tf.fill([self.batch_size], tf.cast(self.go_symbol, tf.int32))
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.decoder_embeddings, start_tokens, self.eos_symbol)

        maximum_iterations = self.params['max_output_len']

        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, self.initial_state,
            output_layer=self.projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)
        
        self.translations = outputs.sample_id


    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.train_loss, var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.train_step = optimizer.apply_gradients(clipped_grads)
        self.sess.run(tf.local_variables_initializer())


    def process_data(self, data, mode):
        restrict = self.args.get("--restrict_data")
        if restrict is not None and int(restrict) > 0:
            data = data[:int(restrict)]

        # Get some common data out:
        if mode == ModeKeys.TRAIN and (self.params['input_shape'] is None or self.params['target_vocab_size'] is None):
            for g in data:
                self.max_input_len = max(self.max_input_len, g["node_features"].shape[0])
                self.max_output_len = max(self.max_output_len, g['output'].shape[0] + 1)
                self.tgt_vocab_size = max(self.tgt_vocab_size, np.max(g['output'])+1 if len(g['output']) > 0 else 0) 
                             
            self.src_vocab_size = max(self.src_vocab_size, data[0]["node_features"].shape[1])
            
            self.output_shape = (self.max_output_len, )
            self.go_symbol  = self.tgt_vocab_size
            self.eos_symbol = self.tgt_vocab_size + 1
            self.pad_symbol = self.tgt_vocab_size + 2

        return self.process_raw_graphs(data, mode)


    def process_raw_graphs(self, raw_data, mode):
        processed_graphs = []
        for d in raw_data:
            processed_graphs.append({"init": d["node_features"]})

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

    def make_minibatch_iterator(self, data, mode):
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
            batch_target_task_values = []
            batch_decoder_inputs = []
            batch_sequence_lens = []
            batch_target_mask = []
            batch_graph_nodes_list = []
            node_offset = 0
            
            final_graph_nodes = self.max_input_len
            while num_graphs < len(data) and node_offset + final_graph_nodes < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = cur_graph['init'].shape[0]
                batch_node_features = vstack([batch_node_features, cur_graph['init']])

                pad_len = self.max_input_len - num_nodes_in_graph
                pad_nodes = csr_matrix((pad_len, cur_graph['init'].shape[1]), dtype=np.int32)
                batch_node_features = vstack([batch_node_features, pad_nodes])
                batch_graph_sizes.append(num_nodes_in_graph)
                num_nodes_in_graph = self.max_input_len

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
                self.placeholders['encoder_inputs']: (indices, batch_node_features.col, (batch_node_features.shape[0], )),
                self.placeholders['encoder_lengths'] : np.array(batch_graph_sizes)
            }

            batch_feed_dict[self.placeholders['decoder_lengths']] = np.array(batch_graph_sizes)
            
            if mode != ModeKeys.INFER:
                batch_feed_dict.update({
                    self.placeholders['decoder_targets']: np.array(batch_target_task_values),
                    self.placeholders['decoder_inputs']: np.array(batch_decoder_inputs),
                    self.placeholders['decoder_lengths']: np.array(batch_sequence_lens),
                    self.placeholders['target_mask'] : np.array(batch_target_mask),
                })




            yield batch_feed_dict

    def run_epoch(self, epoch_name, data, mode):
        loss = 0
        start_time = time.time()
        processed_graphs = 0
        batch_results = []
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, mode), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['encoder_lengths']].shape[0]
            processed_graphs += num_graphs
            if mode == ModeKeys.TRAIN:
                fetch_list = [self.train_loss, self.train_step]
            elif mode == ModeKeys.EVAL:
                fetch_list = [self.train_loss, self.translations]
            elif mode == ModeKeys.INFER:
                fetch_list = self.translations

            result = self.sess.run(fetch_list, feed_dict=batch_data)

            if mode != ModeKeys.INFER:
                loss += result[0] * num_graphs

                print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                                step,
                                                                                num_graphs,
                                                                                loss / processed_graphs),
                                                                                end='\r')
            if mode == ModeKeys.TRAIN:
                results = []
            elif mode == ModeKeys.EVAL:
                results = result[1:]
            elif mode == ModeKeys.INFER:
                results = result

            batch_results.append(results)

        instance_per_sec = processed_graphs / (time.time() - start_time)
        if mode == ModeKeys.INFER:
            return batch_results
        else: 
            loss = loss / processed_graphs
            return loss, batch_results, instance_per_sec

    def train(self, train_data, valid_data):
        # Load data:
        self.max_input_len = 0
        self.max_output_len = 0
        self.tgt_vocab_size = 0
        self.src_vocab_size = 0
        self.train_data = self.process_data(train_data, mode=ModeKeys.TRAIN)
        self.valid_data = self.process_data(valid_data, mode=ModeKeys.EVAL)

        if self.graph is None:
            self.build_model()

        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                valid_loss, valid_results, valid_speed = self.run_epoch("Resumed (validation)", 
                                                                        self.valid_data, ModeKeys.EVAL)

                format_string, valid_log = self.get_log(valid_loss, valid_speed, valid_results,
                                                        ModeKeys.EVAL)

                best_checkpoint_metric = self.get_checkpoint_metric(valid_log)
                best_epoch = 0
                print("\r\x1b[KResumed operation, initial val run:" + format_string % valid_log)
            else:
                (best_checkpoint_metric, best_epoch) = (None, 0)

            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_results, train_speed = self.run_epoch("epoch %i (training)" % epoch,
                                                                        self.train_data, ModeKeys.TRAIN)

                format_string, train_log = self.get_log(train_loss, train_speed, train_results,
                                                        ModeKeys.TRAIN)

                print(("\r\x1b[K Train: " + format_string) % train_log)
                valid_loss, valid_results, valid_speed = self.run_epoch("epoch %i (validation)" % epoch,
                                                                        self.valid_data, ModeKeys.EVAL)


                format_string, valid_log = self.get_log(valid_loss, valid_speed, valid_results,
                                                        ModeKeys.EVAL)

                print(("\r\x1b[K Valid: " + format_string) % valid_log)

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': train_log,
                    'valid_results': valid_log,
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)


                checkpoint_metric = self.get_checkpoint_metric(valid_log)
                if best_checkpoint_metric is None or checkpoint_metric > best_checkpoint_metric:
                    self.save_model(self.best_model_file)
                    print("  (Best epoch so far. Saving to '%s')" % (self.best_model_file))
                    best_checkpoint_metric = checkpoint_metric
                    best_epoch = epoch
                elif epoch - best_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on checkpoint metric" % self.params['patience'])
                    break


    def get_log(self, loss, speed, extra_results, mode):
        if mode == ModeKeys.TRAIN:
            return "loss : %.5f | instances/sec: %.2f", (loss, speed)

        elif mode == ModeKeys.EVAL:
            format_string, metrics = "loss: %.5f", (loss,)

            #Extract sampled sentences from results
            sampled_ids = [sampled_sentence.tolist() for result in extra_results for sampled_sentence in result[0]]

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
            sentences.append(sampled_sentence)
        return sentences

    def save_model(self, path):
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

            data_to_save = {
                            "params": self.params,
                            "weights": weights_to_save
                        }

            with open(path, 'wb') as out_file:
                pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    
    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids or num epochs:
            if par not in ['num_epochs', 'batch_size']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)



            


            


