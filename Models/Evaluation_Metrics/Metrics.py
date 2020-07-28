#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf

def evaluation(y, prediction):
    '''

    This is the evaluation metrics.
    Here, we provide a four-class classification evaluation codes
    The files will be updated to adapt multiply classes soon.

    Notice: All the evaluations during training will be saved in the TensorBoard.

    Use
        tensorboard --logdir="Your_Checkpoint_File_Address" --host=127.0.0.1
    to view and download the evaluation files.

    BTW, you can definitely edit the following codes to adapt your own classes,
    e.g., if Two classes, u can just delete the three and four class.

    Args:
        y: The True Labels (with One-hot representation)
        prediction: The predicted output probability

    Returns:
        Single class evaluation: T1_accuracy, T1_Precision, T1_Recall, T1_F_Score,
                                 T2_accuracy, T2_Precision, T2_Recall, T2_F_Score,
                                 T3_accuracy, T3_Precision, T3_Recall, T3_F_Score,
                                 T4_accuracy, T4_Precision, T4_Recall, T4_F_Score,

        Global model evaluation: Global_Average_Accuracy,
                                 Kappa_Metric,
                                 Macro_Global_Precision,
                                 Macro_Global_Recall,
                                 Macro_Global_F1_Score

    '''

    # Calculate Accuracy
    # Add metrics to TensorBoard.
    with tf.name_scope('Evalution'):
        # Calculate Each Task Accuracy
        with tf.name_scope('Each_Class_accuracy'):
            # Task 1 Accuracy
            with tf.name_scope('T1_accuracy'):
                # Number of Classified Correctly
                y_T1 = tf.equal(tf.argmax(y, 1), 0)
                prediction_T1 = tf.equal(tf.argmax(prediction, 1), 0)
                T1_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T1), tf.float32))

                # Number of All the Test Samples
                T1_all_Num = tf.reduce_sum(tf.cast(y_T1, tf.float32))

                # Task 1 Accuracy
                T1_accuracy = tf.divide(T1_Corrected_Num, T1_all_Num)
                tf.summary.scalar('T1_accuracy', T1_accuracy)

                T1_TP = T1_Corrected_Num
                T1_TN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T1), tf.math.logical_not(prediction_T1)),
                            tf.float32))
                T1_FP = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T1), prediction_T1), tf.float32))
                T1_FN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(y_T1, tf.math.logical_not(prediction_T1)), tf.float32))

                with tf.name_scope("T1_Precision"):
                    T1_Precision = T1_TP / (T1_TP + T1_FP)
                    tf.summary.scalar('T1_Precision', T1_Precision)

                with tf.name_scope("T1_Recall"):
                    T1_Recall = T1_TP / (T1_TP + T1_FN)
                    tf.summary.scalar('T1_Recall', T1_Recall)

                with tf.name_scope("T1_F_Score"):
                    T1_F_Score = (2 * T1_Precision * T1_Recall) / (T1_Precision + T1_Recall)
                    tf.summary.scalar('T1_F_Score', T1_F_Score)

            # Task 2 Accuracy
            with tf.name_scope('T2_accuracy'):
                # Number of Classified Correctly
                y_T2 = tf.equal(tf.argmax(y, 1), 1)
                prediction_T2 = tf.equal(tf.argmax(prediction, 1), 1)
                T2_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T2), tf.float32))

                # Number of All the Test Samples
                T2_all_Num = tf.reduce_sum(tf.cast(y_T2, tf.float32))

                # Task 2 Accuracy
                T2_accuracy = tf.divide(T2_Corrected_Num, T2_all_Num)
                tf.summary.scalar('T2_accuracy', T2_accuracy)

                T2_TP = T2_Corrected_Num
                T2_TN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T2), tf.math.logical_not(prediction_T2)),
                            tf.float32))
                T2_FP = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T2), prediction_T2), tf.float32))
                T2_FN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(y_T2, tf.math.logical_not(prediction_T2)), tf.float32))

                with tf.name_scope("T2_Precision"):
                    T2_Precision = T2_TP / (T2_TP + T2_FP)
                    tf.summary.scalar('T2_Precision', T2_Precision)

                with tf.name_scope("T2_Recall"):
                    T2_Recall = T2_TP / (T2_TP + T2_FN)
                    tf.summary.scalar('T2_Recall', T2_Recall)

                with tf.name_scope("T2_F_Score"):
                    T2_F_Score = (2 * T2_Precision * T2_Recall) / (T2_Precision + T2_Recall)
                    tf.summary.scalar('T2_F_Score', T2_F_Score)

            # Task 3 Accuracy
            with tf.name_scope('T3_accuracy'):
                # Number of Classified Correctly
                y_T3 = tf.equal(tf.argmax(y, 1), 2)
                prediction_T3 = tf.equal(tf.argmax(prediction, 1), 2)
                T3_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T3), tf.float32))

                # Number of All the Test Samples
                T3_all_Num = tf.reduce_sum(tf.cast(y_T3, tf.float32))

                # Task 3 Accuracy
                T3_accuracy = tf.divide(T3_Corrected_Num, T3_all_Num)
                tf.summary.scalar('T3_accuracy', T3_accuracy)

                T3_TP = T3_Corrected_Num
                T3_TN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T3), tf.math.logical_not(prediction_T3)),
                            tf.float32))
                T3_FP = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T3), prediction_T3), tf.float32))
                T3_FN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(y_T3, tf.math.logical_not(prediction_T3)), tf.float32))

                with tf.name_scope("T3_Precision"):
                    T3_Precision = T3_TP / (T3_TP + T3_FP)
                    tf.summary.scalar('T3_Precision', T3_Precision)

                with tf.name_scope("T3_Recall"):
                    T3_Recall = T3_TP / (T3_TP + T3_FN)
                    tf.summary.scalar('T3_Recall', T3_Recall)

                with tf.name_scope("T3_F_Score"):
                    T3_F_Score = (2 * T3_Precision * T3_Recall) / (T3_Precision + T3_Recall)
                    tf.summary.scalar('T3_F_Score', T3_F_Score)

            # Task 4 Accuracy
            with tf.name_scope('T4_accuracy'):
                # Number of Classified Correctly
                y_T4 = tf.equal(tf.argmax(y, 1), 3)
                prediction_T4 = tf.equal(tf.argmax(prediction, 1), 3)
                T4_Corrected_Num = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T4), tf.float32))

                # Number of All the Test Samples
                T4_all_Num = tf.reduce_sum(tf.cast(y_T4, tf.float32))

                # Task 4 Accuracy
                T4_accuracy = tf.divide(T4_Corrected_Num, T4_all_Num)
                tf.summary.scalar('T4_accuracy', T4_accuracy)

                T4_TP = T4_Corrected_Num
                T4_TN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T4), tf.math.logical_not(prediction_T4)),
                            tf.float32))
                T4_FP = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(tf.math.logical_not(y_T4), prediction_T4), tf.float32))
                T4_FN = tf.reduce_sum(
                    tf.cast(tf.math.logical_and(y_T4, tf.math.logical_not(prediction_T4)), tf.float32))

                with tf.name_scope("T4_Precision"):
                    T4_Precision = T4_TP / (T4_TP + T4_FP)
                    tf.summary.scalar('T4_Precision', T4_Precision)

                with tf.name_scope("T4_Recall"):
                    T4_Recall = T4_TP / (T4_TP + T4_FN)
                    tf.summary.scalar('T4_Recall', T4_Recall)

                with tf.name_scope("T4_F_Score"):
                    T4_F_Score = (2 * T4_Precision * T4_Recall) / (T4_Precision + T4_Recall)
                    tf.summary.scalar('T4_F_Score', T4_F_Score)

        # Calculate the Confusion Matrix
        with tf.name_scope("Confusion_Matrix"):
            with tf.name_scope("T1_Label"):
                T1_T1 = T1_Corrected_Num
                T1_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T2), tf.float32))
                T1_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T3), tf.float32))
                T1_T4 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T1, prediction_T4), tf.float32))

                T1_T1_percent = tf.divide(T1_T1, T1_all_Num)
                T1_T2_percent = tf.divide(T1_T2, T1_all_Num)
                T1_T3_percent = tf.divide(T1_T3, T1_all_Num)
                T1_T4_percent = tf.divide(T1_T4, T1_all_Num)

                tf.summary.scalar('T1_T1_percent', T1_T1_percent)
                tf.summary.scalar('T1_T2_percent', T1_T2_percent)
                tf.summary.scalar('T1_T3_percent', T1_T3_percent)
                tf.summary.scalar('T1_T4_percent', T1_T4_percent)

            with tf.name_scope("T2_Label"):
                T2_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T1), tf.float32))
                T2_T2 = T2_Corrected_Num
                T2_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T3), tf.float32))
                T2_T4 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T2, prediction_T4), tf.float32))

                T2_T1_percent = tf.divide(T2_T1, T2_all_Num)
                T2_T2_percent = tf.divide(T2_T2, T2_all_Num)
                T2_T3_percent = tf.divide(T2_T3, T2_all_Num)
                T2_T4_percent = tf.divide(T2_T4, T2_all_Num)

                tf.summary.scalar('T2_T1_percent', T2_T1_percent)
                tf.summary.scalar('T2_T2_percent', T2_T2_percent)
                tf.summary.scalar('T2_T3_percent', T2_T3_percent)
                tf.summary.scalar('T2_T4_percent', T2_T4_percent)

            with tf.name_scope("T3_Label"):
                T3_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T1), tf.float32))
                T3_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T2), tf.float32))
                T3_T3 = T3_Corrected_Num
                T3_T4 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T3, prediction_T4), tf.float32))

                T3_T1_percent = tf.divide(T3_T1, T3_all_Num)
                T3_T2_percent = tf.divide(T3_T2, T3_all_Num)
                T3_T3_percent = tf.divide(T3_T3, T3_all_Num)
                T3_T4_percent = tf.divide(T3_T4, T3_all_Num)

                tf.summary.scalar('T3_T1_percent', T3_T1_percent)
                tf.summary.scalar('T3_T2_percent', T3_T2_percent)
                tf.summary.scalar('T3_T3_percent', T3_T3_percent)
                tf.summary.scalar('T3_T4_percent', T3_T4_percent)

            with tf.name_scope("T4_Label"):
                T4_T1 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T1), tf.float32))
                T4_T2 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T2), tf.float32))
                T4_T3 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_T4, prediction_T3), tf.float32))
                T4_T4 = T4_Corrected_Num

                T4_T1_percent = tf.divide(T4_T1, T4_all_Num)
                T4_T2_percent = tf.divide(T4_T2, T4_all_Num)
                T4_T3_percent = tf.divide(T4_T3, T4_all_Num)
                T4_T4_percent = tf.divide(T4_T4, T4_all_Num)

                tf.summary.scalar('T4_T1_percent', T4_T1_percent)
                tf.summary.scalar('T4_T2_percent', T4_T2_percent)
                tf.summary.scalar('T4_T3_percent', T4_T3_percent)
                tf.summary.scalar('T4_T4_percent', T4_T4_percent)

        with tf.name_scope('Global_Evalution_Metrics'):
            # Global Average Accuracy - Simple Algorithm
            with tf.name_scope('Global_Average_Accuracy'):
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                Global_Average_Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('Global_Average_Accuracy', Global_Average_Accuracy)

            with tf.name_scope('Kappa_Metric'):
                Test_Set_Num = T1_all_Num + T2_all_Num + T3_all_Num + T4_all_Num

                Actual_T1 = T1_all_Num
                Actual_T2 = T2_all_Num
                Actual_T3 = T3_all_Num
                Actual_T4 = T4_all_Num

                Prediction_T1 = T1_T1 + T2_T1 + T3_T1 + T4_T1
                Prediction_T2 = T1_T2 + T2_T2 + T3_T2 + T4_T2
                Prediction_T3 = T1_T3 + T2_T3 + T3_T3 + T4_T3
                Prediction_T4 = T1_T4 + T2_T4 + T3_T4 + T4_T4

                p0 = (T1_T1 + T2_T2 + T3_T3 + T4_T4) / Test_Set_Num
                pe = (Actual_T1 * Prediction_T1 + Actual_T2 * Prediction_T2 + Actual_T3 * Prediction_T3 + Actual_T4 * Prediction_T4) / \
                     (Test_Set_Num * Test_Set_Num)

                Kappa_Metric = (p0 - pe) / (1 - pe)
                tf.summary.scalar('Kappa_Metric', Kappa_Metric)

            with tf.name_scope('Micro_Averaged_Evalution'):
                with tf.name_scope("Micro_Averaged_Confusion_Matrix"):
                    TP_all = T1_TP + T2_TP + T3_TP + T4_TP
                    TN_all = T1_TN + T2_TN + T3_TN + T4_TN
                    FP_all = T1_FP + T2_FP + T3_FP + T4_FP
                    FN_all = T1_FN + T2_FN + T3_FN + T4_FN

                with tf.name_scope("Micro_Global_Precision"):
                    Micro_Global_Precision = TP_all / (TP_all + FP_all)
                    tf.summary.scalar('Micro_Global_Precision', Micro_Global_Precision)

                with tf.name_scope("Micro_Global_Recall"):
                    Micro_Global_Recall = TP_all / (TP_all + FN_all)
                    tf.summary.scalar('Micro_Global_Recall', Micro_Global_Recall)

                with tf.name_scope("Micro_Global_F1_Score"):
                    Micro_Global_F1_Score = (2 * Micro_Global_Precision * Micro_Global_Recall) / (
                            Micro_Global_Precision + Micro_Global_Recall)
                    tf.summary.scalar('Micro_Global_F1_Score', Micro_Global_F1_Score)

            with tf.name_scope('Macro_Averaged_Evalution'):
                with tf.name_scope("Macro_Global_Precision"):
                    Macro_Global_Precision = (T1_Precision + T2_Precision + T3_Precision + T4_Precision) / 4
                    tf.summary.scalar('Macro_Global_Precision', Macro_Global_Precision)

                with tf.name_scope("Macro_Global_Recall"):
                    Macro_Global_Recall = (T1_Recall + T2_Recall + T3_Recall + T4_Recall) / 4
                    tf.summary.scalar('Macro_Global_Recall', Macro_Global_Recall)

                with tf.name_scope("Macro_Global_F1_Score"):
                    Macro_Global_F1_Score = (T1_F_Score + T2_F_Score + T3_F_Score + T4_F_Score) / 4
                    tf.summary.scalar('Macro_Global_F1_Score', Macro_Global_F1_Score)

    # You DON'T have to return these because all the criterias have been saved in the TensorBoard
    # return T1_accuracy, T1_Precision, T1_Recall, T1_F_Score, \
    #        T2_accuracy, T2_Precision, T2_Recall, T2_F_Score, \
    #        T3_accuracy, T3_Precision, T3_Recall, T3_F_Score, \
    #        T4_accuracy, T4_Precision, T4_Recall, T4_F_Score, \
    #        Global_Average_Accuracy, Kappa_Metric, \
    #        Macro_Global_Precision, Macro_Global_Recall, Macro_Global_F1_Score

    # Instead, you can only return Accuracy Criteria to compare the capacity of your Model
    return Global_Average_Accuracy
