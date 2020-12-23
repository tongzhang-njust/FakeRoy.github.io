'''
base_model.py
'''

import tensorflow as tf
import numpy as np

class BaseModel(object):

    def __init__(self):
        self.regularizers = []

    def _weight_variable(self, shape, regularization = True):

        initial = tf.truncated_normal_initializer(0,0.1)
        var  = tf.get_variable('weights',shape, tf.float32, initializer = initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.histogram_summary(var.op.name, var)

        return var
   
    def _bias_variable(self, shape, regularization = True):
        
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer = initial) 
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.histogram_summary(var.op.name, var)

        return var

    def _cross_entropy_loss(self, logits, labels, regularization):

        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization


        ## summaries for tensorboard
        tf.scalar_summary('loss/cross_entropy', cross_entropy)
        tf.scalar_summary('loss/regularization', regularization)
        tf.scalar_summary('loss/total', loss)
        with tf.name_scope('averages'):
            averages = tf.train.ExponentialMovingAverage(0.9)
            op_averages = averages.apply([cross_entropy, regularization, loss])
            tf.scalar_summary('loss/avg/cross_entropy', averages.average(cross_entropy))
            tf.scalar_summary('/loss/avg/regularization', averages.average(regularization))
            tf.scalar_summary('/loss/avg/loss', averages.average(loss))
            with tf.control_dependencies([op_averages]):
                loss_average = tf.identity(averages.average(loss), name='control')

        return loss, loss_average

    def _euclidean_loss(self, y_pred, y, regularization):

        with tf.name_scope('loss'):
            with tf.name_scope('euclidean_dist'):
                #labels = tf.to_int64(labels)
                #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
                cross_entropy = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y)))
                #cross_entropy = tf.reduce_mean((y_pred-y))
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization


        ## summaries for tensorboard
        tf.scalar_summary('loss/euclidean_dist', cross_entropy)
        tf.scalar_summary('loss/regularization', regularization)
        tf.scalar_summary('loss/total', loss)
        with tf.name_scope('averages'):
            averages = tf.train.ExponentialMovingAverage(0.9)
            op_averages = averages.apply([cross_entropy, regularization, loss])
            tf.scalar_summary('loss/avg/euclidean_dist', averages.average(cross_entropy))
            tf.scalar_summary('/loss/avg/regularization', averages.average(regularization))
            tf.scalar_summary('/loss/avg/loss', averages.average(loss))
            with tf.control_dependencies([op_averages]):
                loss_average = tf.identity(averages.average(loss), name='control')

        return loss, loss_average



   
 




