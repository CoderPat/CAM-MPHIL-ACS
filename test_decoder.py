import numpy as np
import tensorflow as tf

g = [[1, 1, 0, 0], [0, 0, 1, 1]]
v = [[1, 2, 3], [2,3,4],[3,4,5], [4,5,6]]

ph1 = tf.placeholder(tf.int32, shape=[None, None])
ph2 = tf.placeholder(tf.float32, shape=[None, 3])
padded_nodes = tf.concat([tf.zeros([1, tf.shape(ph2)[1]]), ph2], axis=0)

zero = tf.constant(0, dtype=tf.int32, shape=[2, 4])
where = tf.not_equal(ph1, zero)

indices = tf.where(where)
mask = tf.sparse_to_dense(indices, [2, 4], indices[:, 1] + 1)

sess = tf.Session()
print(sess.run(tf.nn.embedding_lookup(padded_nodes, mask, None), feed_dict={ph1:g, ph2:v}))

