#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf


# Initialize the Weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)

    return tf.Variable(initial)


# Initialize the Bias
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)

    return tf.Variable(initial)
