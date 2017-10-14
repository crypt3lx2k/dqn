#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import argparse
import os
import sys

import gym

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from config import config
from config import get_FLAGS
from graph import build_graph
from model import DQNModel

import dqn

FLAGS = None

def visualize_filter (x):
    # HWCN -> CNHW
    x = np.transpose(x, [2, 3, 0, 1])

    # Normalize to [0, 1]
    x_min = x.min()
    x_max = x.max()

    x = (x-x_min)/(x_max-x_min)

    # Pad in between filter images
    m, n, h, w = x.shape

    x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

    x = np.transpose(x, [0, 2, 1, 3])
    x = np.reshape(x, [(h+2)*m, (w+2)*n])

    # Show images to screen
    plt.imshow(x,cmap='hot')

def main (_):
    # Initialize all variables
    if not os.path.exists(FLAGS.checkpoint_path):
        print('No model at {}'.format(FLAGS.checkpoint_path), file=sys.stderr)
        return 1

    graph = build_graph (
        dqn.relevant_actions,
        optimizer=FLAGS.optimizer,
        stacked_shape=FLAGS.stacked_shape,
        learning_rate=FLAGS.learning_rate
    )

    model = DQNModel (
        graph,
        FLAGS.checkpoint_path,
        FLAGS.target_update_frequency,
        FLAGS.checkpoint_frequency
    )

    # Get all weights of action value network
    tvars = graph.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'action_value')

    # Extract only the convolutional filters
    filters = filter(lambda v : 'conv' in v.name and 'kernel' in v.name, tvars)
    filters = model.session.run(filters)
    filters_n = len(filters)

    for i, x in enumerate(filters):
        visualize_filter(x)
        plt.show()

    return 0

if __name__ == '__main__':
    FLAGS, unparsed = get_FLAGS(config)
    dqn.FLAGS = FLAGS

    exit(main([sys.argv[0]] + unparsed))
