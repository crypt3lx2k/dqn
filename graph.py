#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf

from config import AttrDict

optimizer_function = lambda lr : tf.train.RMSPropOptimizer(lr, momentum=0.95, epsilon=0.01, centered=True)

def build_model (x, output_n, training=False):
    net = x

    # 3 input conv layers batch_norm is experimental
    net = tf.layers.conv2d(net, 16, (9, 9), (4, 4), padding='VALID', name='conv_0')
    net = tf.layers.batch_normalization(net, training=training, name='conv_0/norm')
    net = tf.nn.relu(net, name='conv_0/relu')

    net = tf.layers.conv2d(net, 32, (7, 7), (2, 2), padding='VALID', name='conv_1')
    net = tf.layers.batch_normalization(net, training=training, name='conv_1/norm')
    net = tf.nn.relu(net, name='conv_1/relu')

    net = tf.layers.conv2d(net, 64, (5, 5), (2, 2), padding='VALID', name='conv_2')
    net = tf.layers.batch_normalization(net, training=training, name='conv_2/norm')
    net = tf.nn.relu(net, name='conv_2/relu')

    # Flat activations
    dims = np.prod(net.shape.as_list()[1:])
    net = tf.reshape(net, [-1, dims], name='activations_flat')

    # Dropout in training model, also experimental
    net = tf.layers.dropout(net, rate=0.5, training=training, name='hidden_0/dropout')
    net = tf.layers.dense(net, 512, activation=tf.nn.relu, name='hidden_0')
    net = tf.layers.dense(net, output_n, activation=None, use_bias=True, name='predictions')

    return net

def build_action_value (x, a, t, action_n):
    # Create inference model
    with tf.variable_scope('model'):
        predictions = build_model(x, action_n)
        value_estimates = tf.reduce_max(predictions,axis=-1)

    # Create training model
    with tf.variable_scope('model', reuse=True):
        predictions_training = build_model(x, action_n, training=True)

    with tf.variable_scope('loss'):
        # Mask out actions that were not taken from loss
        index_mask = tf.one_hot (
            a, action_n, name='action_index_mask'
        )

        t = tf.expand_dims(t, axis=-1, name='targets/expanded')

        # ||r + gamma*max Q(s') - Q(s)||^2
        loss = tf.losses.mean_squared_error (
            labels=index_mask*t, predictions=predictions_training, weights=index_mask
        )

    return AttrDict ({
        'predictions' : predictions,
        'value_estimates' : value_estimates,
        'loss' : loss
    })

def build_graph (observation_shape, action_shape, action_n, learning_rate):
    graph = tf.Graph()

    with graph.as_default():
        with tf.variable_scope('inputs'):
            x = tf.placeholder (
                dtype=tf.uint8,
                shape=(None,) + observation_shape,
                name='observations'
            )

            a = tf.placeholder (
                dtype=tf.int32,
                shape=(None,),
                name='actions'
            )

            t = tf.placeholder (
                dtype=tf.float32,
                shape=(None,),
                name='targets'
            )

            # Scale and convert image to float while on the gpu
            image = tf.image.convert_image_dtype (
                x, tf.float32, name='observation_image'
            )

        with tf.variable_scope('action_value'):
            action_value = build_action_value(image, a, t, action_n)

        with tf.variable_scope('target_value'):
            target_value = build_action_value(image, a, t, action_n)

            assignment_operators = []
            for action_variable, target_variable in zip (
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'action_value'),
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_value')
            ):
                assign = tf.assign(target_variable, action_variable)
                assignment_operators.append(assign)

            target_value.update_weights = tf.group(*assignment_operators)

        with tf.variable_scope('train'):
            total_loss = action_value.loss
            global_step = tf.train.get_or_create_global_step()

            optimizer = optimizer_function(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train = optimizer.minimize(total_loss, global_step=global_step)

        with tf.variable_scope('init'):
            initialize = tf.global_variables_initializer()

        with tf.variable_scope('saver'):
            saver = tf.train.Saver(save_relative_paths=True)

    graph.finalize()

    return AttrDict ({
        'x' : x,
        'a' : a,
        't' : t,
        'action_value' : action_value,
        'target_value' : target_value,
        'train' : train,
        'global_step' : global_step,
        'initialize' : initialize,
        'saver' : saver,
        'graph' : graph
    })
