import numpy as np
import tensorflow as tf

g = [[1, 1, 1, 0], [0, 0, 1, 1]]
v = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [-4, -5, -6]]

ph1 = tf.placeholder(tf.int32, shape=[None, None])
ph2 = tf.placeholder(tf.float32, shape=[None, 3])

emb = (tf.cast(ph1, tf.float32) @ ph2) / tf.reshape(tf.cast(tf.reduce_sum(ph1, 1), tf.float32), (-1, 1))

sess = tf.Session()
print(sess.run(emb, feed_dict={ph1:g, ph2:v}))