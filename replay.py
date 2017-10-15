#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
from numpy.lib.stride_tricks import as_strided

class Experience (object):
    def __init__ (self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.shape = (capacity,)
        self.state_shape = self.shape + observation_shape

        frame_channels = self.state_shape[-1]
        self.storage_shape = self.shape + observation_shape[:-1] + (frame_channels+1,)

        # Initialize memory, due to overcommitted memory you can allocate more
        # than the free memory here. Change allocator from np.empty to np.ones
        # if you need guarantees ahead of time.
        self.actions = np.empty(self.shape + action_shape, dtype=np.int8)
        self.rewards = np.empty(self.shape, dtype=np.int8)
        self.terminals = np.empty(self.shape, dtype=np.bool)

        # For stacked frame storage we utilize the fact that observation and successor
        # may at most differ by 1 frame, so we store a stack of frame_channels+1 and
        # provide strided view into this storage.
        self.storage = np.empty(self.storage_shape, dtype=np.uint8)

        self.observations = as_strided (
            self.storage[...,0], shape=self.state_shape, strides=self.storage.strides
        )
        self.successors = as_strided (
            self.storage[...,1], shape=self.state_shape, strides=self.storage.strides
        )

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
        self.successors[self.index,...,-1] = successor[...,-1]

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
