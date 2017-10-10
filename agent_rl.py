#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np

class Schedule (object):
    pass

class ConstantSchedule (Schedule):
    def __init__ (self, epsilon=0.0):
        self.epsilon = epsilon

    def __call__ (self, *args, **kwargs):
        return self.epsilon

class LinearSchedule (Schedule):
    def __init__ (self, start=None, rest=None, steps=None):
        self.start = start
        self.rest = rest
        self.steps = steps

    def __call__ (self, step):
        if step >= self.steps:
            return self.rest

        b = self.start
        a = (self.rest - self.start)/self.steps

        return a * step + b

class Update (object):
    pass

class TDZeroUpdate (Update):
    def __init__ (self, model, batch_size, gamma):
        self.model = model
        self.batch_size = batch_size
        self.gamma = gamma

    def __call__ (self, memory):
        # Acquire mini-batch and run value estimates
        states, actions, rewards, successors, terminals = memory.get_batch(self.batch_size)
        value_estimates = self.model.value_estimates(successors)

        # Set terminal value estimates to 0
        value_estimates[terminals] = 0.0

        # TD-target
        targets = rewards + self.gamma * value_estimates

        # Perform update
        loss, step = self.model.update(states, actions, targets)
        return loss, step

class Policy (object):
    pass

class EpsilonGreedyPolicy (Policy):
    def __init__ (self, model, schedule, n_actions):
        self.model = model
        self.epsilon = None

        self.schedule = schedule
        self.n_actions = n_actions

        # Initialize self.epsilon to correct value
        self.update(model.get_step())

    def __call__ (self, state):
        # Returns random action with some probability epsilon otherwise
        # act greedily with respect to the Q-values
        draw = np.random.rand()

        # Random action with probability epsilon
        if draw < self.epsilon:
            return np.random.choice(self.n_actions)

        # Otherwise we pick the action that maximizes expected return
        state = state.reshape((1,) + state.shape)
        predictions = self.model.predict(state)

        return np.argmax(predictions)

    def update (self, step):
        self.epsilon = self.schedule(step)
        return self.epsilon

class Agent (object):
    def perceive (self, state, reward, terminal):
        raise NotImplementedError()

class DQNAgent (Agent):
    def __init__ (
            self,
            preprocessor=None,
            memory=None,
            policy=None, update=None,
            actions_per_update=None,
            replay_start_size=None
    ):
        # Class instances
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.update = update

        # Hyper parameters
        self.actions_per_update = actions_per_update
        self.replay_start_size = replay_start_size

        # Attributes
        self.counter = 0

    def act (self, state):
        # Get state on a form fit for policy
        self.state = self.preprocessor(state)

        # Get action from policy
        action = self.policy(self.state)

        # Increment action counter
        self.counter += 1

        return action

    def perceive (self, state, action, reward, terminal):
        # Store copy of variables overwritten by self.act
        original_state = self.state.copy()
        original_action = action

        if terminal:
            self.preprocessor.reset()

        action = self.act(state)

        # Record observation
        if self.memory is not None:
            self.memory.record(original_state, original_action, reward, self.state, terminal)

            # Do update if necessary
            if ((self.memory.stored >= self.replay_start_size) and
                (self.counter % self.actions_per_update == 0)):

                loss, step = self.update(self.memory)
                epsilon = self.policy.update(step)

                if step % 100 == 0:
                    print('Batch {} loss={}'.format(step, loss))

        return action
