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

from config import config
from graph import build_graph
from model import DQNModel

import agent_rl
import dqn
import frame_processing

def main (_):
    # Initialize all variables
    environment = gym.make(FLAGS.environment)

    if not os.path.exists(FLAGS.checkpoint_path):
        print('No model at {}'.format(FLAGS.checkpoint_path), file=sys.stderr)
        return 1

    graph = build_graph (
        FLAGS.stacked_shape,
        FLAGS.action_shape,
        len(dqn.relevant_actions),
        FLAGS.learning_rate
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
    evaluation_update = agent_rl.DoNothingUpdate()

    evaluation_agent = agent_rl.DQNAgent (
        preprocessor=stacking_preprocessor,
        policy=evaluation_policy, update=evaluation_update
    )

    for episode in xrange(100):
        wins, losses = dqn.run_episode (
            environment, evaluation_agent,
            render=False, delay=0.0
        )
        print('Evaluation at {}! wins={}, losses={}'.format(episode, wins, losses))

    environment.close()

    return 0

if __name__ == '__main__':
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

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.action_shape = ()
    FLAGS.frame_shape = (FLAGS.frame_height, FLAGS.frame_width)
    FLAGS.stacked_shape = FLAGS.frame_shape + (FLAGS.frame_channels,)

    dqn.FLAGS = FLAGS
    exit(main([sys.argv[0]] + unparsed))
