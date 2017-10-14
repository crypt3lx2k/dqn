#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf

from graph import build_graph

class Model (object):
    pass

class DQNModel (Model):
    def __init__ (self, graph, model_dir, target_update_frequency, checkpoint_frequency):
        self.graph = graph
        self.model_dir = model_dir

        self.target_update_frequency = target_update_frequency
        self.checkpoint_frequency = checkpoint_frequency

        self.session = tf.Session(graph=self.graph.graph)

        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)

        if latest_checkpoint:
            self.graph.saver.restore(self.session, latest_checkpoint)
            print('restored from {}'.format(latest_checkpoint))
        else:
            self.session.run(self.graph.initialize)
            print('initialized model')

        step = self.get_step()

        self.last_target_update_step = step
        self.last_checkpoint_step = step

    def predict (self, observation):
        return self.session.run (
            self.graph.action_value.predictions,
            feed_dict = {self.graph.x : observation}
        )

    def value_estimates (self, observations):
        return self.session.run (
            self.graph.target_value.value_estimates,
            feed_dict = {self.graph.x : observations}
        )

    def target_update (self, step):
        _ = self.session.run (
            self.graph.target_value.update_weights
        )

        self.last_target_update_step = step

    def save_checkpoint (self, step):
        path = self.graph.saver.save (
            self.session, self.model_dir + '/model',
            global_step=self.graph.global_step
        )

        print('Saved checkpoint to {} at {}'.format(path, step))
        self.last_checkpoint_step = step

    def update (self, states, actions, targets):
        step, loss, _ = self.session.run (
            [
                self.graph.global_step,
                self.graph.action_value.loss,
                self.graph.train
            ],
            feed_dict = {
                self.graph.x : states,
                self.graph.a : actions,
                self.graph.t : targets
            }
        )

        if step >= (self.last_target_update_step + self.target_update_frequency):
            self.target_update(step)

        if step >= (self.last_checkpoint_step + self.checkpoint_frequency):
            self.save_checkpoint(step)

        return step, loss

    def get_step (self):
        return self.session.run(self.graph.global_step)
