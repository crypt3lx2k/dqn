#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import argparse
import os
import time
import sys

import gym

import numpy as np

from config import config
from graph import build_graph
from model import DQNModel

import replay
import frame_processing

relevant_actions = [0, 2, 3]

FLAGS = None

def get_epsilon_greedy_actor (model, epsilon):
    def actor (environment, observation):
        # Returns random action with some probability epsilon otherwise
        # act greedily with respect to the Q-values
        draw = np.random.rand()

        # Random action with probability epsilon
        if draw < epsilon:
            return np.random.choice(len(relevant_actions))

        # Otherwise we pick the action that maximizes expected return
        observation = observation.reshape((1,) + observation.shape)
        predictions = model.predict(observation)

        return np.argmax(predictions)

    return actor

def get_update (model, memory):
    def update ():
        # If we're below the start limit just skip this update
        if memory.stored < FLAGS.replay_start_size:
            return 0

        # Acquire mini-batch and run value estimates
        states, actions, rewards, successors, terminals = memory.get_batch(FLAGS.batch_size)
        value_estimates = model.value_estimates(successors)

        # Set terminal value estimates to 0
        value_estimates[terminals] = 0.0

        # TD-target
        targets = rewards + FLAGS.gamma * value_estimates

        loss, step = model.update(states, actions, targets)
        if step % 100 == 0:
            print('Batch {} loss={}'.format(step, loss))

        return step

    return update

def run_episode (environment, preprocessor, actor, memory=None, update=None, render=False, delay=0.0):
    # Set up environment
    frame = environment.reset()
    successor = preprocessor(frame)

    game_step = 0
    done = False
    terminal = False

    actions = 0
    observation = None

    wins = 0
    losses = 0

    while not done:
        if render:
            environment.render()

        get_action = (game_step % FLAGS.repeats_per_action) == 0

        # Record memory
        if memory is not None and (get_action or terminal) and observation is not None:
            stored = memory.record(observation, action, reward, successor, terminal)

        # Get action from actor
        if get_action:
            # Update observation with successor
            observation = successor.copy()

            # Acquire action
            action = actor(environment, observation)
            actions += 1

        # Step with action
        frame, reward, done, info = environment.step(relevant_actions[action])
        successor = preprocessor(frame)

        terminal = (reward != 0.0) or done

        # Overwrite successor if terminal
        if terminal:
            preprocessor.reset()

            wins += reward == 1.0
            losses += reward == -1.0

        # Do update
        if update is not None and actions % FLAGS.actions_per_update == 0:
            update()

        if delay:
            time.sleep(delay)

        game_step += 1

    return wins, losses

def epsilon_schedule (step):
    if step >= FLAGS.epsilon_steps:
        return FLAGS.epsilon_rest

    b = FLAGS.epsilon_start
    a = (FLAGS.epsilon_rest - FLAGS.epsilon_start)/FLAGS.epsilon_steps

    return a * step + b

def main (_):
    # Initialize all variables
    environment = gym.make(FLAGS.environment)

    if not os.path.exists(FLAGS.checkpoint_path):
        print('Making directory path {}'.format(FLAGS.checkpoint_path))
        os.makedirs(FLAGS.checkpoint_path)

    graph = build_graph (
        FLAGS.stacked_shape,
        FLAGS.action_shape,
        len(relevant_actions),
        FLAGS.learning_rate
    )

    model = DQNModel (
        graph,
        FLAGS.checkpoint_path,
        FLAGS.target_update_frequency,
        FLAGS.checkpoint_frequency
    )

    memory = replay.Experience(FLAGS.replay_capacity, FLAGS.stacked_shape, FLAGS.action_shape)

    greedy = get_epsilon_greedy_actor(model, epsilon=FLAGS.epsilon_greedy)
    update = get_update(model, memory)

    stacking_preprocessor = frame_processing.StackingPreprocessor(FLAGS.stacked_shape)

    for episode in xrange(1000000):
        step = model.get_step()

        # Evaluate quality with a greedy policy
        if episode % 100 == 0:
            wins, losses = run_episode (
                environment, stacking_preprocessor, greedy,
                render=False, delay=0.0
            )
            print('Evaluation at {}! wins={}, losses={}'.format(episode, wins, losses))
            # environment.render(close=True)

        # Linear annealing schedule for epsilon
        epsilon = epsilon_schedule(step)
        actor = get_epsilon_greedy_actor(model, epsilon=epsilon)

        # Policy rollouts
        wins, losses = run_episode (
            environment, stacking_preprocessor, actor,
            memory=memory, update=update, render=False
        )

        print('Episode {} done! epsilon={}, wins={}, losses={}'.format(episode, epsilon, wins, losses))
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

    exit(main([sys.argv[0]] + unparsed))
