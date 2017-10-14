#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import sys

import gym

import numpy as np

from config import config
from config import get_FLAGS
from graph import build_graph
from model import DQNModel

import agent_rl
import dqn
import frame_processing

FLAGS = None

def main (_):
    # Initialize all variables
    environment = gym.make(FLAGS.environment)

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

    stacking_preprocessor = frame_processing.StackingPreprocessor(FLAGS.stacked_shape)

    # Set up evaluation agent
    evaluation_schedule = agent_rl.ConstantSchedule(epsilon=FLAGS.epsilon_greedy)
    evaluation_policy = agent_rl.EpsilonGreedyPolicy(model, evaluation_schedule, dqn.relevant_actions_n)

    evaluation_agent = agent_rl.DQNAgent (
        preprocessor=stacking_preprocessor,
        policy=evaluation_policy
    )

    for episode in xrange(100):
        wins, losses = dqn.run_episode (
            environment, evaluation_agent,
            render=FLAGS.render, delay=FLAGS.delay
        )
        print('Evaluation at {}! wins={}, losses={}'.format(episode, wins, losses))

    environment.close()

    return 0

if __name__ == '__main__':
    FLAGS, unparsed = get_FLAGS(config)
    dqn.FLAGS = FLAGS

    exit(main([sys.argv[0]] + unparsed))
