#!/usr/bin/env/python

from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf
import queue
import threading
import collections
import math

SMALL_NUMBER = 1e-7

# ---- INITIALIZATION ----

def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

# ---- METRICS ----- 

def compute_bleu(references, translations, max_order=4):
    """
    Computes BLEU for a evaluation set of translations
    Based on https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
    """

    def get_ngrams(segment, max_order):
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (reference, translation) in zip(references, translations):
        reference_length += len(reference)
        translation_length += len(translation)

        reference_ngram_counts = get_ngrams(reference, max_order)
        translation_ngram_counts = get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & reference_ngram_counts

        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
            precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = translation_length / reference_length

    if ratio > 1.0:
        bp = 1.
    elif ratio == 0:
        bp = 0
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu


def compute_f1(references, translations, beta=1, unk_token=None):
    """
    Computes BLEU for a evaluation set of translations
    Based on https://github.com/mast-group/convolutional-attention/blob/master/convolutional_attention/f1_evaluator.pyy
    """

    total_f1 = 0
    for (reference, translation) in zip(references, translations):
        tp = 0
        ref = list(reference)
        for token in set(translation):
            if token == unk_token:
                continue  
            if token in ref:
                ref.remove(token)
                tp += 1

        if len(translation) > 0:
            precision = tp / len(translation)
        else:
            precision = 0

        if len(reference) > 0:
            recall = tp / len(reference)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = (1+beta**2) * precision * recall / ((beta**2 * precision) + recall)
        else:
            f1 = 0

        total_f1 += f1
    return total_f1 / len(translations)


# ---- VECTORIZER --------

class Vectorizer:
    def __init__(self, minimum_frequency=5, min_sentence_freq=2):
        self.__minimum_frequency = minimum_frequency
        self.__min_sentence_freq = min_sentence_freq
        self.__dictionary = None
        self.__reversed_dictionary = None
    
    def build_and_vectorize(self, corpus, subtokens=True):
        self.__dictionary = {}
        all_counts = defaultdict(lambda: 0)
        distinct_counts = defaultdict(lambda: 0)
        for sentence in corpus:
            seen_tokens = set()
            def process_term(term):
                if term not in seen_tokens:
                    distinct_counts[term] += 1
                    seen_tokens.add(term)
                all_counts[term] += 1
                if term not in self.__dictionary and \
                    all_counts[term] >= self.__minimum_frequency and \
                    distinct_counts[term] >= self.__min_sentence_freq:
                    self.__dictionary[term] = len(self.__dictionary)+1

            if subtokens:
                for term_dict in sentence:
                    for term in term_dict.keys():
                        process_term(term)

            else:
                for term in sentence:
                    process_term(term)

        self.__reversed_dictionary = dict(zip(self.__dictionary.values(), self.__dictionary.keys()))
        return self.vectorize(corpus, subtokens=subtokens)
        
    def vectorize(self, corpus, subtokens=True):
        if subtokens:
            vectors = []
            for sentence in corpus:
                indices, indptr, values = ([], [0], [])
                for term_dict in sentence:
                    for term, frequency in term_dict.items():
                        indices.append(self.__dictionary[term] if term in self.__dictionary else 0)
                        values.append(frequency)
                    indptr.append(indptr[-1]+len(term_dict))
                vectors.append(csr_matrix((values, indices, indptr), (len(sentence), len(self.__dictionary)+1)))
            return vectors
        else:
            return [[self.__dictionary[term] if term in self.__dictionary else 0 for term in sentence] for sentence in corpus]


    def devectorize(self, vectors, subtokens=True):
        all_terms = []
        if subtokens:
            for vector in vectors:
                i = 0
                terms = []
                for ptr in vector.indptr[1:]:
                    label = []
                    while i < ptr and i < len(vector.indices):
                        label.append(self.__reversed_dictionary[vector.indices[i]] if vector.indices[i] > 0 else 'UNK')
                        i += 1
                    terms.append("_".join(label))
                all_terms.append(terms)
        else:
            for vector in vectors:
                terms = []
                for index in vector:
                    terms.append(self.__reversed_dictionary[index] if index > 0 else 'UNK')
                all_terms.append(terms)
        return all_terms

#--- Threaded iterator for minibatch handling ---

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
