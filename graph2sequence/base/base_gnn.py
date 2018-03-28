#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random

from tensorflow.contrib.learn import ModeKeys

from .utils import ThreadedIterator, SMALL_NUMBER

class BaseGNN(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.0005,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 200,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'random_seed': 0,

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

    def process_data(self, data, mode):
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
            self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))
            self.output_shape = np.array(data[0]['output']).shape

        return self.process_raw_graphs(data, mode)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [None] + list(self.output_shape),
                                                            name='target_values')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            self.ops['final_node_representations'] = self.compute_final_node_representations()
            if not self.params['use_graph']:
                self.ops['final_node_representations'] = tf.zeros_like(self.ops['final_node_representations'])

        computed_outputs = self.build_output(self.ops['final_node_representations'])
        self.ops['loss'] = self.get_loss(computed_outputs, self.placeholders['target_values'])
        self.training_ops = self.get_training_ops(computed_outputs, self.placeholders['target_values'])
        self.validation_ops = self.get_validation_ops(computed_outputs, self.placeholders['target_values'])
        self.inference_ops = self.get_inference_ops(computed_outputs)


    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def run_epoch(self, epoch_name: str, data, mode):
        loss = 0
        start_time = time.time()
        processed_graphs = 0
        batch_results = []
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, mode), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if mode == ModeKeys.TRAIN:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], self.ops['train_step']] + self.training_ops
            elif mode == ModeKeys.EVAL:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss']] + self.validation_ops
            elif mode == ModeKeys.INFER:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = self.inference_ops

            result = self.sess.run(fetch_list, feed_dict=batch_data)

            if mode != ModeKeys.INFER:
                loss += result[0] * num_graphs

                print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                                   step,
                                                                                   num_graphs,
                                                                                   loss / processed_graphs),
                                                                                   end='\r')

            # Submodels handle the results so we just pass them along
            if mode == ModeKeys.TRAIN:
                results = result[2:]
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
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
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

    def infer(self, input_data):
        input_data = self.process_data(input_data, mode=ModeKeys.INFER)

        with self.graph.as_default():
            infer_results = self.run_epoch(None, input_data, ModeKeys.INFER)

        return self.process_inference(infer_results)
    

    def save_model(self, path: str) -> None:
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

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids or num epochs:
            if par not in ['task_ids', 'num_epochs']:
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


    # Template methods that normally subclasses should redefine
    # -----------------------------------------------------------------------

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, mode: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def build_output(self, last_h) -> tf.Tensor:
        raise Exception("Models have to implement layers after hidden node representation!")

    def get_loss(self, computed_outputs, target_outputs) -> tf.Tensor:
        raise Exception("")

    def get_training_ops(self, computed_outputs, target_outputs):
        return []
    
    def get_evaluation_ops(self, computed_outputs, target_outputs):
        return []

    def get_inference_ops(self, computed_outputs):
        raise Exception("Models have to specify inference ops")

    def get_log(self, loss, speed, results, mode):
        return "loss : %.5f | instances/sec: %.2f", (loss, speed)

    def get_checkpoint_metric(self, valid_log):
        return -valid_log[0]

    def process_inference(self, inference_results):
        raise Exception("Models have to implement process_inference")