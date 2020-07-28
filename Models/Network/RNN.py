#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf


def RNN(Input, max_time, n_input, rnn_size, keep_prob, weights_1, biases_1, weights_2, biases_2):
    '''

    Args:
        Input: The reshaped input EEG signals
        max_time: The unfolded time slice of RNN Model
        n_input: The input signal size at one time
        rnn_size: The number of RNN units inside the RNN Model
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer

    Returns:
        FC_2: Final prediction of RNN Model
        FC_1: Extracted features from the first fully connected layer

    '''

    # One layer RNN Model
    Input = tf.reshape(Input, [-1, max_time, n_input])
    rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
    rnn_drop = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, input_keep_prob=keep_prob)
    outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_drop, inputs=Input, dtype=tf.float32)

    # First fully-connected layer
    FC_1 = tf.matmul(final_state, weights_1) + biases_1
    FC_1 = tf.layers.batch_normalization(FC_1, training=True)
    FC_1 = tf.nn.softplus(FC_1)
    FC_1 = tf.nn.dropout(FC_1, keep_prob)

    # Second fully-connected layer
    FC_2 = tf.matmul(FC_1, weights_2) + biases_2
    FC_2 = tf.nn.softmax(FC_2)

    return FC_2, FC_1

