#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import time
import sys

import gym

import numpy as np

from config import config
from config import get_FLAGS
from graph import build_graph
from model import DQNModel

import agent_rl
import replay
import frame_processing

relevant_actions = [0, 2, 3]
relevant_actions_n = len(relevant_actions)

FLAGS = None

def run_episode (environment, agent, render=False, delay=0.0):
    # Set up environment
    state = environment.reset()
    action = agent.act(state)

    done = False

    wins = 0
    losses = 0

    while not done:
        if render:
            environment.render()

        # Act upon environment
        state, reward, done, info = environment.step(relevant_actions[action])
        terminal = (reward != 0.0) or done

        if terminal:
            wins += (reward == 1.0)
            losses += (reward == -1.0)

        # Get next action from agent
        action = agent.perceive(state, action, reward, terminal)

        if render and delay:
            time.sleep(delay)

    return wins, losses

def main (_):
    # Initialize all variables
    environment = gym.make(FLAGS.environment)

    if not os.path.exists(FLAGS.checkpoint_path):
        print('Making directory path {}'.format(FLAGS.checkpoint_path))
        os.makedirs(FLAGS.checkpoint_path)

    graph = build_graph (
        relevant_actions,
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

    memory = replay.Experience(FLAGS.replay_capacity, FLAGS.stacked_shape, FLAGS.action_shape)
    stacking_preprocessor = frame_processing.StackingPreprocessor(FLAGS.stacked_shape)

    # Set up evaluation agent
    evaluation_schedule = agent_rl.ConstantSchedule(epsilon=FLAGS.epsilon_greedy)
    evaluation_policy = agent_rl.EpsilonGreedyPolicy(model, evaluation_schedule, relevant_actions_n)

    evaluation_agent = agent_rl.DQNAgent (
        preprocessor=stacking_preprocessor,
        policy=evaluation_policy
    )

    # Set up learning agent
    learning_schedule = agent_rl.LinearSchedule (
        start=FLAGS.epsilon_start,
        rest=FLAGS.epsilon_rest,
        steps=FLAGS.epsilon_steps
    )
    learning_policy = agent_rl.EpsilonGreedyPolicy(model, learning_schedule, relevant_actions_n)
    learning_update = agent_rl.TDZeroUpdate(model, FLAGS.batch_size, FLAGS.gamma)

    learning_agent = agent_rl.DQNAgent (
        preprocessor=stacking_preprocessor,
        memory=memory,
        policy=learning_policy, update=learning_update,
        actions_per_update=FLAGS.actions_per_update,
        replay_start_size=FLAGS.replay_start_size
    )

    # Run episodes
    for episode in xrange(1000000):
        # Evaluate quality with a greedy policy
        if episode % 100 == 0:
            wins, losses = run_episode (
                environment, evaluation_agent,
                render=FLAGS.render, delay=FLAGS.delay
            )
            print('Evaluation at {}! wins={}, losses={}'.format(episode, wins, losses))

        # Policy rollouts
        wins, losses = run_episode (
            environment, learning_agent
        )

        print('Episode {} done! epsilon={}, wins={}, losses={}'.format(episode, learning_agent.policy.epsilon, wins, losses))
    environment.close()

    return 0

if __name__ == '__main__':
    FLAGS, unparsed = get_FLAGS(config)

    exit(main([sys.argv[0]] + unparsed))
