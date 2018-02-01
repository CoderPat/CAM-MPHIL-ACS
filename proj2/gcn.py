
import tensorflow as tf
import scipy.sparse as sp 
import numpy as np

from sklearn.metrics import accuracy_score
from data.utils import load_data

def xavier(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    l = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(l)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



class GCN:
    def __init__(self, epochs=200, hidden_size=16, learning_rate=0.01, weight_decay=5e-4, optimizer='adam'):
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)


    def build_model(self, input_shape, output_dim):
        self.input = tf.placeholder(tf.float32, shape=input_shape)
        self.mask = tf.placeholder(tf.int32, shape=(input_shape[0],))
        self.normalized_adj = tf.sparse_placeholder(tf.float32)
        self.labels = tf.placeholder(tf.int32, shape=(input_shape[0], output_dim))
        self.dropout = tf.placeholder_with_default(0., shape=[], name=None)

        self.hidden_vars, hidden_layer = self.build_gcn_layer(self.input, self.normalized_adj, 
                                                              self.hidden_size, activation=tf.nn.relu,
                                                              dropout=self.dropout)
        __, output_layer = self.build_gcn_layer(hidden_layer, self.normalized_adj, output_dim, dropout=self.dropout)

        self.score_layer = output_layer


    def build_gcn_layer(self, input_tensor, normalized_adj, output_dim, 
                        dropout=None, name="gcn_layer", activation=lambda x: x):
        if dropout is not None:
            input_tensor = tf.nn.dropout(input_tensor, 1-dropout)

        input_dim = input_tensor.get_shape()[-1].value
        weights = xavier([input_dim, output_dim],name='weights')
        bias = tf.zeros([output_dim], name='bias')

        output = tf.matmul(input_tensor, weights)
        output = tf.sparse_tensor_dense_matmul(normalized_adj, output) 
        return [weights, bias], activation(output) 

    
    def loss(self, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        mask = tf.cast(self.mask, dtype=tf.float32)
        mask = mask / tf.reduce_mean(mask)
        loss = tf.reduce_mean(loss * mask)
        for var in self.hidden_vars:
            loss += self.weight_decay * tf.nn.l2_loss(var)
        return loss

    def accuracy(self, logits):
        errors = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        accuracy = tf.cast(errors, tf.float32)
        mask = tf.cast(self.mask, dtype=tf.float32)
        mask = mask / tf.reduce_mean(mask)
        accuracy = accuracy * mask
        return tf.reduce_mean(accuracy)

    def fit(self, X, Y, adj_matrix, mask):
        input_shape = X.shape
        output_dim = Y.shape[1]
        normalized_adj = normalize_adj(adj_matrix)
        normalized_input = normalize_features(X)
        self.build_model(input_shape, output_dim)
        self.loss_t = self.loss(self.score_layer)
        self.accuracy_t = self.accuracy(self.score_layer)
        train_op = self.optimizer.minimize(self.loss_t)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        for epoch in range(self.epochs):
            feed_dict = {self.input: normalized_input, 
                         self.mask: mask,
                         self.normalized_adj: normalized_adj,
                         self.labels : Y,
                         self.dropout: 0.5}
            _, loss, accuracy = self.session.run([train_op, self.loss_t, self.accuracy_t], feed_dict=feed_dict)
            print("Epoch: %04d, train_loss=%5f, train_acc=%5f" %
                  (epoch, loss, accuracy))

    def predict(self, X, adj_matrix):
        normalized_adj = normalize_adj(adj_matrix)
        normalized_input = normalize_features(X)
        feed_dict = {self.input: normalized_input, 
                     self.normalized_adj: normalized_adj}
        scores = self.session.run(self.score_layer, feed_dict)
        print(np.array(scores).shape)
        labels = np.argmax(scores, axis=1)
        return labels

    def evaluate(self, X_test, Y_test, adj_matrix, test_mask):
        normalized_adj = normalize_adj(adj_matrix)
        normalized_input = normalize_features(X_test)
        feed_dict = {self.input: normalized_input, 
                     self.normalized_adj: normalized_adj,
                     self.mask: test_mask,
                     self.labels: Y_test}
        return self.session.run(self.accuracy_t, feed_dict)



if __name__ == "__main__":
    adj, X, Y_train, Y_val, Y_test, train_mask, val_mask, test_mask = load_data("citeseer")
    model = GCN()
    model.fit(X, Y_train, adj, train_mask)
    print(model.evaluate(X, Y_test, adj, test_mask))

    

