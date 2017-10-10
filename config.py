#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

class AttrDict (dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

config = AttrDict ({
    'environment' : 'Pong-v0',

    'batch_size' : 32,
    'gamma' : 0.99,

    'repeats_per_action' : 4,
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

    'learning_rate' : 1e-4
})

config.frame_shape = (config.frame_height, config.frame_width)
config.stacked_shape = config.frame_shape + (config.frame_channels,)
