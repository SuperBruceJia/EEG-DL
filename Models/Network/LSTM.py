#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf


def LSTM(Input, max_time, n_input, lstm_size, keep_prob, weights_1, biases_1, weights_2, biases_2):
    '''

    Args:
        Input: The reshaped input EEG signals
        max_time: The unfolded time slice of LSTM Model
        n_input: The input signal size at one time
        rnn_size: The number of LSTM units inside the LSTM Model
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer

    Returns:
        FC_2: Final prediction of LSTM Model
        FC_1: Extracted features from the first fully connected layer

    '''

    # One layer RNN Model
    Input = tf.reshape(Input, [-1, max_time, n_input])
    cell_encoder = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
    encoder_drop = tf.contrib.rnn.DropoutWrapper(cell=cell_encoder, input_keep_prob=keep_prob)
    outputs_encoder, final_state_encoder = tf.nn.dynamic_rnn(cell=encoder_drop, inputs=Input, dtype=tf.float32)

    # First fully-connected layer
    # final_state_encoder[0] is the long-term memory
    FC_1 = tf.matmul(final_state_encoder[0], weights_1) + biases_1
    FC_1 = tf.layers.batch_normalization(FC_1, training=True)
    FC_1 = tf.nn.softplus(FC_1)
    FC_1 = tf.nn.dropout(FC_1, keep_prob)

    # Second fully-connected layer
    FC_2 = tf.matmul(FC_1, weights_2) + biases_2
    FC_2 = tf.nn.softmax(FC_2)

    return FC_2, FC_1

