#!/usr/bin/env/python

from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf
import queue
import threading

SMALL_NUMBER = 1e-7

def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

class Vectorizer:
    def __init__(self, minimum_frequency=2):
        self.__minimum_frequency = minimum_frequency
        self.__dictionary = None
        self.__reversed_dictionary = None
    
    def build_and_vectorize(self, term_dics):
        self.__dictionary = {}
        counts = defaultdict(lambda: 0)
        for term_dict in term_dics:
            for term in term_dict.keys():
                counts[term] += 1
                if counts[term] == self.__minimum_frequency:
                    self.__dictionary[term] = len(self.__dictionary)+1

        self.__reversed_dictionary = dict(zip(self.__dictionary.values(), self.__dictionary.keys()))
        return self.vectorize(term_dics)
        
    def vectorize(self, term_dics, indices_only=False):
        indices, indptr, values = ([], [0], [])
        for term_dict in term_dics:
            for term, frequency in term_dict.items():
                indices.append(self.__dictionary[term] if term in self.__dictionary else 0)
                values.append(frequency)
            indptr.append(indptr[-1]+len(term_dict))
        
        if indices_only:
            return indices
        else:
            return csr_matrix((values, indices, indptr), (len(term_dics), len(self.__dictionary)+1))

    def devectorize(self, vectors, indices_only=False):
        term_dicts = []
        if indices_only:
            for index in vectors:
                term_dicts.append({self.__reversed_dictionary[index] if index > 0 else 'UNK': 1})
        return term_dicts
            


class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int=2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden
