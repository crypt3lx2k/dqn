#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import argparse

class AttrDict (dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

config = AttrDict ({
    'environment' : 'Pong-v0',

    'batch_size' : 32,
    'gamma' : 0.99,

    # Frame skipping is implemented in the Pong-v0 env
    'repeats_per_action' : 1,
    'actions_per_update' : 4,

    'epsilon_start' : 1.0,
    'epsilon_rest' : 0.1,
    'epsilon_steps' : 100000,
    'epsilon_greedy' : 0.00,

    'replay_capacity' : 50000,
    'replay_start_size' : 5000,

    'target_update_frequency' : 128,

    'checkpoint_frequency' : 10000,
    'checkpoint_path' : '/tmp/atari/pong',

    'frame_height' : 84,
    'frame_width' : 84,
    'frame_channels' : 4,

    'optimizer' : 'Adam',
    'learning_rate' : 1e-4,

    'render' : False,
    'delay' : 0.0,

    # Defined beneath
    'frame_shape' : None,
    'stacked_shape' : None
})

config.frame_shape = (config.frame_height, config.frame_width)
config.stacked_shape = config.frame_shape + (config.frame_channels,)

def get_FLAGS (config):
    parser = argparse.ArgumentParser()

    parser.add_argument (
        '--environment', type=str, metavar='env',
        default=config.environment,
        help='OpenAI gym environment.'
    )

    parser.add_argument (
        '--batch-size', type=int, metavar='n',
        default=config.batch_size,
        help='Number of SARS to use in each update.'
    )

    parser.add_argument (
        '--gamma', type=float, metavar='x',
        default=config.gamma,
        help='Discount factor.'
    )

    parser.add_argument (
        '--learning-rate', type=float, metavar='x',
        default=config.learning_rate,
        help='Learning rate.'
    )

    parser.add_argument (
        '--optimizer', type=str, metavar='opt',
        default=config.optimizer,
        help='Tensorflow.train optimizer to use.'
    )

    parser.add_argument (
        '--checkpoint-path', type=str, metavar='dir',
        default=config.checkpoint_path,
        help='Where to store model files.'
    )

    parser.add_argument (
        '--frame-height', type=int, metavar='n',
        default=config.frame_height,
        help='Target height for input frames.'
    )

    parser.add_argument (
        '--frame-width', type=int, metavar='n',
        default=config.frame_width,
        help='Target width for input frames.'
    )

    parser.add_argument (
        '--frame-channels', type=int, metavar='n',
        default=config.frame_channels,
        help='Number of consecutive frames to stack for network input.'
    )

    parser.add_argument (
        '--target-update-frequency', type=int, metavar='n',
        default=config.target_update_frequency,
        help='Number of updates between each synchronization between training and target networks.'
    )

    parser.add_argument (
        '--checkpoint-frequency', type=int, metavar='n',
        default=config.checkpoint_frequency,
        help='Number of updates between storing checkpoint on disk.'
    )

    parser.add_argument (
        '--epsilon-start', type=float, metavar='eps',
        default=config.epsilon_start,
        help='Start value for exploration probability.'
    )

    parser.add_argument (
        '--epsilon-rest', type=float, metavar='eps',
        default=config.epsilon_rest,
        help='Stop value for exploration probability.'
    )

    parser.add_argument (
        '--epsilon-greedy', type=float, metavar='eps',
        default=config.epsilon_greedy,
        help='Exploration value used in greedy-evaluation policy.'
    )

    parser.add_argument (
        '--epsilon-steps', type=int, metavar='n',
        default=config.epsilon_steps,
        help='Number of training steps before exploration probability reaches rest value.'
    )

    parser.add_argument (
        '--repeats-per-action', type=int, metavar='n',
        default=config.repeats_per_action,
        help='How many times an action is repeated before we get a new one from our policy.'
    )

    parser.add_argument (
        '--actions-per-update', type=int, metavar='n',
        default=config.actions_per_update,
        help='How many actions we get from policy before each update.'
    )

    parser.add_argument (
        '--replay-capacity', type=int, metavar='n',
        default=config.replay_capacity,
        help='Total number of SARS in replay memory.'
    )

    parser.add_argument (
        '--replay-start-size', type=int, metavar='n',
        default=config.replay_start_size,
        help='Number of elements in replay memory before we start updating.'
    )

    parser.add_argument (
        '--render',
        action='store_false' if config.render else 'store_true',
        help='Render evaluation runs to screen.'
    )

    parser.add_argument (
        '--delay', type=float, metavar='t',
        default=config.delay,
        help='Seconds to delay between each frame in evaluation renders..'
    )

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.action_shape = ()
    FLAGS.frame_shape = (FLAGS.frame_height, FLAGS.frame_width)
    FLAGS.stacked_shape = FLAGS.frame_shape + (FLAGS.frame_channels,)

    FLAGS.delay = 0.0 if not FLAGS.render else FLAGS.delay

    return FLAGS, unparsed
