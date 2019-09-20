# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

n_features = 4
n_neurons = 5
n_timesteps = 2

#The data is a sequence of a number from 0 to 9 and divided into three batches of data.
## Data 
X_batch = np.array([
                    [[0, 1, 2, 5], 
                     [9, 8, 7, 4]], # Batch 1
                    
                    [[3, 4, 5, 2], 
                     [0, 0, 0, 0]], # Batch 2
                    
                    [[6, 7, 8, 5], 
                     [6, 5, 4, 2]], # Batch 3
                                     ])


tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_timesteps, n_features], name = "X")


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)		
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)	


init = tf.global_variables_initializer()
with tf.Session() as sess:
    
#    train_writer = tf.summary.FileWriter( './logs_tf', sess.graph)
    
    init.run()
    outputs_val = sess.run(outputs, feed_dict={X: X_batch})

    print(states.eval(feed_dict={X: X_batch}))
    print("==================================")
    print(outputs.eval(feed_dict={X: X_batch}))

#[[ 0.38941205 -0.9980438   0.99750966  0.7892596   0.9978241   0.9999997 ]
# [ 0.61096436  0.7255889   0.82977575 -0.88226104  0.29261455 -0.15597084]
# [ 0.62091285 -0.87023467  0.99729395 -0.58261937  0.9811445   0.99969864]]