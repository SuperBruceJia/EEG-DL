#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf


def loss(y, prediction, l2_norm=True):
    '''

    This is the Loss Function (Euclidean Distance)
    We will provide more functions later.

    Args:
        y: The true label
        prediction: predicted label
        l2_norm: l2 regularization, set True by default

    Returns:
        loss: The loss of the Model

    '''
    train_variable = tf.trainable_variables()

    if l2_norm == False:
        loss = tf.reduce_mean(tf.square(y - prediction))

    else:
        norm_coefficient = 0.001
        regularization_loss = norm_coefficient * tf.reduce_sum([tf.nn.l2_loss(v) for v in train_variable])
        model_loss = tf.reduce_mean(tf.square(y - prediction))
        loss = tf.reduce_mean(model_loss + regularization_loss)

    return loss
