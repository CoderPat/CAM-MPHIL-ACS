import numpy as np
import tensorflow as tf


def unflatten_nodes(graphs, nodes):

  
  _, _, counts = tf.unique_with_counts(indices[:,0])
  num_elems = tf.shape(counts)[0]
  init_array = tf.TensorArray(tf.int32, size=num_elems)
  def loop_body(i, ta):
    return i + 1, ta.write(i, tf.range(counts[i]))

  _, result_array = tf.while_loop(
    lambda i, ta: i < num_elems, loop_body, [0, init_array])

  return tf.transpose(tf.concat([[indices[:, 0]], [result_array.concat()]], axis=0))


g = [[1, 1, 1, 0], [0, 0, 1, 1]]
v = [[1, 2, 3], [2,3,4], [3,4,5], [-4,-5,-6]]

ph1 = tf.placeholder(tf.int32, shape=[None, None])
ph2 = tf.placeholder(tf.float32, shape=[None, 3])

padded_nodes = tf.concat([tf.zeros([1, tf.shape(ph2)[1]]), ph2], axis=0)

zero = tf.constant(0, dtype=tf.int32, shape=[2, 4])
indices = tf.cast(tf.where(tf.not_equal(ph1, zero)), tf.int32)

shifted_indices = shift_indices(indices)
mask = tf.sparse_to_dense(shifted_indices, [2, 3], indices[:, 1] + 1)
states = tf.nn.embedding_lookup(padded_nodes, mask, None)

sess = tf.Session()
print(sess.run([states], feed_dict={ph1:g, ph2:v}))

"""
def unflatten_nodes(graphs, all_nodes):
    num_graphs = self.placeholders['num_graphs']

    padded_nodes = tf.concat([tf.zeros([1, tf.shape(nodes)[1]]), nodes], axis=0) 
    zero = tf.constant(0, dtype=tf.int32, shape=[2, 4])
    indices = tf.cast(tf.where(tf.not_equal(ph1, zero)), tf.int32) # indices of [g x v]
    
    _, _, counts = tf.unique_with_counts(indices[:,0])
    num_elems = tf.shape(counts)[0]
    init_array = tf.TensorArray(tf.int32, size=num_elems)

    def loop_body(i, ta):
        return i + 1, ta.write(i, tf.range(counts[i]))

    _, result_array = tf.while_loop(
        lambda i, ta: i < num_elems, loop_body, [0, init_array])

    shifted_indices = \                                          
        tf.transpose(tf.concat([[indices[:, 0]], [result_array.concat()]], axis=0)) # indices of [g x max_v]

    mask = tf.sparse_to_dense(shifted_indices, [num_graphs, self.max_num_vertices], indices[:, 1] + 1)
    return tf.nn.embedding_lookup(padded_nodes, mask, None)
"""