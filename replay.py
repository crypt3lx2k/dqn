#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np

class Experience (object):
    def __init__ (self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.shape = (capacity,)
        self.state_shape = self.shape + observation_shape

        # Initialize memory, due to overcommitted memory you can allocate more
        # than the free memory here. Change allocator from np.empty to np.ones
        # if you need guarantees ahead of time.
        self.observations = np.empty(self.state_shape, dtype=np.uint8)
        self.successors = np.empty(self.state_shape, dtype=np.uint8)
        self.actions = np.empty(self.shape + action_shape, dtype=np.int8)
        self.rewards = np.empty(self.shape, dtype=np.int8)
        self.terminals = np.empty(self.shape, dtype=np.bool)

        # Index points to current open slot and will wrap around
        self.index = self.capacity
        # Stored refers to total number of stored elements
        self.stored = 0

    def record (self, observation, action, reward, successor, terminal):
        # Wrap around if full
        if self.index >= self.capacity:
            self.index = 0

        # Store SARS in open slot
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.successors[self.index] = successor

        # Also keep track of terminal states
        self.terminals[self.index] = terminal

        # Increment open slot index
        self.index += 1

        # Increment total store counter if under capacity
        if self.stored < self.capacity:
            self.stored += 1

        return self.stored

    def get_batch (self, batch_size):
        # Pick batch_size random elements from stored memory
        # Note that np.random.choice has the ability pick the
        # same element multiple times in one batch.
        indices = np.random.choice(self.stored, batch_size)

        # Return matching SARS
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.successors[indices],
            self.terminals[indices]
        )
