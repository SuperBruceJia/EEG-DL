#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Hide the Configuration and Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import random
import numpy as np
import tensorflow as tf
from Models.DatasetAPI.DataLoader import DatasetLoader
from Models.Network.GRU import GRU
from Models.Loss_Function.Loss import loss
from Models.Evaluation_Metrics.Metrics import evaluation

# Model Name
Model = 'Gated_Recurrent_Unit'

# Clear all the stack and use GPU resources as much as possible
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Your Dataset Location, for example EEG-Motor-Movement-Imagery-Dataset
# The CSV file should be named as training_set.csv, training_label.csv, test_set.csv, and test_label.csv
DIR = 'DatasetAPI/EEG-Motor-Movement-Imagery-Dataset/'
SAVE = 'Saved_Files/' + Model + '/'
if not os.path.exists(SAVE):  # If the SAVE folder doesn't exist, create one
    os.mkdir(SAVE)

# Load the dataset, here it uses one-hot representation for labels
train_data, train_labels, test_data, test_labels = DatasetLoader(DIR=DIR)
train_labels = tf.one_hot(indices=train_labels, depth=4)
train_labels = tf.squeeze(train_labels).eval(session=sess)
test_labels = tf.one_hot(indices=test_labels, depth=4)
test_labels = tf.squeeze(test_labels).eval(session=sess)

# Model Hyper-parameters
n_input  = 64   # The input size of signals at each time
max_time = 64   # The unfolded time slices of the GRU Model
gru_size = 256  # The number of GRUs inside the GRU Model

n_class   = 4     # The number of classification classes
n_hidden  = 64    # The number of hidden units in the first fully-connected layer
num_epoch = 300   # The number of Epochs that the Model run
keep_rate = 0.75  # Keep rate of the Dropout

lr = tf.constant(1e-4, dtype=tf.float32)  # Learning rate
lr_decay_epoch = 50    # Every (50) epochs, the learning rate decays
lr_decay       = 0.50  # Learning rate Decay by (50%)

batch_size = 1024
n_batch = train_data.shape[0] // batch_size

# Initialize Model Parameters (Network Weights and Biases)
# This Model only uses Two fully-connected layers, and u sure can add extra layers DIY
weights_1 = tf.Variable(tf.truncated_normal([gru_size, n_hidden], stddev=0.01))
biases_1  = tf.Variable(tf.constant(0.01, shape=[n_hidden]))
weights_2 = tf.Variable(tf.truncated_normal([n_hidden, n_class], stddev=0.01))
biases_2  = tf.Variable(tf.constant(0.01, shape=[n_class]))

# Define Placeholders
x = tf.placeholder(tf.float32, [None, 64 * 64])
y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)

# Load Model Network
prediction, features = GRU(Input=x,
                           max_time=max_time,
                           n_input=n_input,
                           gru_size=gru_size,
                           keep_prob=keep_prob,
                           weights_1=weights_1,
                           biases_1=biases_1,
                           weights_2=weights_2,
                           biases_2=biases_2)

# Load Loss Function
loss = loss(y=y, prediction=prediction, l2_norm=True)

# Load Optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Load Evaluation Metrics
Global_Average_Accuracy = evaluation(y=y, prediction=prediction)

# Merge all the summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(SAVE + '/train_Writer', sess.graph)
test_writer = tf.summary.FileWriter(SAVE + '/test_Writer')

# Initialize all the variables
sess.run(tf.global_variables_initializer())
for epoch in range(num_epoch + 1):
    # U can use learning rate decay or not
    # Here, we set a minimum learning rate
    # If u don't want this, u definitely can modify the following lines
    learning_rate = sess.run(lr)
    if epoch % lr_decay_epoch == 0 and epoch != 0:
        if learning_rate <= 1e-6:
            lr = lr * 1.0
            sess.run(lr)
        else:
            lr = lr * lr_decay
            sess.run(lr)

    # Randomly shuffle the training dataset and train the Model
    for batch_index in range(n_batch):
        random_batch = random.sample(range(train_data.shape[0]), batch_size)
        batch_xs = train_data[random_batch]
        batch_ys = train_labels[random_batch]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: keep_rate})

    # Show Accuracy and Loss on Training and Test Set
    # Here, for training set, we only show the result of first 100 samples
    # If u want to show the result on the entire training set, please modify it.
    train_accuracy, train_loss = sess.run([Global_Average_Accuracy, loss], feed_dict={x: train_data[0:100], y: train_labels[0:100], keep_prob: 1.0})
    Test_summary, test_accuracy, test_loss = sess.run([merged, Global_Average_Accuracy, loss], feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    test_writer.add_summary(Test_summary, epoch)

    # Show the Model Capability
    print("Iter " + str(epoch) + ", Testing Accuracy: " + str(test_accuracy) + ", Training Accuracy: " + str(train_accuracy))
    print("Iter " + str(epoch) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(train_loss))
    print("Learning rate is ", learning_rate)
    print('\n')

    # Save the prediction and labels for testing set
    # The "labels_for_test.csv" is the same as the "test_label.csv"
    # We will use the files to draw ROC CCurve and AUC
    if epoch == num_epoch:
        output_prediction = sess.run(prediction, feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
        np.savetxt(SAVE + "prediction_for_test.csv", output_prediction, delimiter=",")
        np.savetxt(SAVE + "labels_for_test.csv", test_labels, delimiter=",")

    # if you want to extract and save the features from fully-connected layer, use all the dataset and uncomment this.
    # All data is the total data = training data + testing data
    # We use the features from the overall dataset
    # ML models might be used to classify the features further
    # if epoch == num_epoch:
    #     Features = sess.run(features, feed_dict={x: all_data, y: all_labels, keep_prob: 1.0})
    #     np.savetxt(SAVE + "Features.csv", features, delimiter=",")

train_writer.close()
test_writer.close()
sess.close()
