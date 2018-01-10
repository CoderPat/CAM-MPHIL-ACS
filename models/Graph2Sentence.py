from SparseGNN import SparseGGNN
import tensorflow as tf

class Graph2Sequence:
    def __init__(self, params):
        self.__node_features = tf.placeholder(tf.float32, [None, params['hidden_size']], name='node_features')
        self.__target_values = tf.placeholder(tf.float32, [None, None], name='targets') #MAX OUTPUT LEN?
        self.__num_graphs = tf.placeholder(tf.int64, [], name='num_graphs')
        self.__dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')

        self.__vocab_size = params['vocab_size']
        self.__sos_id = params['sos_id']
        self.__eof_id = parmas['eof_id']

        self.__gnn = SparseGGNN(params)
        num_edge_types = params['n_edge_types']
        self.__adjacency_lists = [tf.placeholder(tf.int64, [None, 2], name='adjacency_e%s' % e)
                                  for e in range(num_edge_types)]
        self.__num_incoming_edges_per_type = tf.placeholder(tf.float32, [None, num_edge_types],
                                                            name='num_incoming_edges_per_type')

        self.__graph_nodes_list = tf.placeholder(tf.int64, [None, 2], name='graph_nodes_list')

        self.__decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_size'])

        self.__loss, self.__accuracy = self.get_loss()

        self.__optimizer = tf.train.AdamOptimizer()

        grads_and_vars = self.__optimizer.compute_gradients(self.__loss)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.train_op = self.__optimizer.apply_gradients(clipped_grads)

    @property
    def node_features(self):
        return self.__node_features

    @property
    def dropout_keep_prob(self):
        return self.__dropout_keep_prob

    @property
    def num_graphs(self):
        return self.__num_graphs

    @property
    def target_values(self):
        return self.__target_values

    @property
    def adjacency_lists(self):
        return self.__adjacency_lists

    @property
    def num_incoming_edges_per_type(self):
        return self.__num_incoming_edges_per_type

    @property
    def graph_nodes_list(self):
        return self.__graph_nodes_list

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy

    def get_logits(self, last_h):
        # calculates graph-level hidden state by summing over vector 
        num_nodes = tf.shape(gate_input, out_type=tf.int64)[0]
        graph_nodes = tf.SparseTensor(indices=self.__graph_nodes_list,
                                      values=tf.ones_like(self.__graph_nodes_list[:, 0], dtype=tf.float32),
                                      dense_shape=[self.__num_graphs, num_nodes])  # [g x v]
                                      
        encoder_state = tf.sparse_tensor_dense_matmul(graph_nodes, last_h) # [g x h]

        projection_layer = layers_core.Dense(self.__vocab_size, use_bias=False)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([self.__num_graphs], self.__sos_id), self.__eos_id)
        decoder = tf.contrib.seq2seq.BasicDecoder(self.__decoder_cell, helper, encoder_state, output_layer=projection_layer)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        logits = outputs.rnn_output
        return logits
        
    def get_loss(self):

        node_representations = self.__gnn.sparse_gnn_layer(self.__node_features,
                                                           self.__adjacency_lists,
                                                           self.__num_incoming_edges_per_type)
        logits = self.get_logits(node_representations)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
        train_loss = (tf.reduce_sum(crossent * target_weights) / self.__num_graphs)
        accuracy = None
        return loss, accuracy

