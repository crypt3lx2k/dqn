#! /usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from scipy.misc import imresize

import numpy as np

luminance_rgb = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)

def preprocess_frame (I, target_shape):
    I = imresize(I, target_shape, interp='bilinear')
    I = I.dot(luminance_rgb)

    return I.astype(np.uint8)

class Preprocessor (object):
    def reset (self):
        raise NotImplementError()

    def __call__ (self, image):
        raise NotImplementError()

class StackingPreprocessor (Preprocessor):
    def __init__ (self, shape, frame_processor=preprocess_frame):
        self.shape = shape
        self.stacked = None
        self.frame_processor = frame_processor

        self.reset()

    def reset (self):
        self.stacked = np.zeros(self.shape, dtype=np.uint8)
        return self

    def __call__ (self, frame):
        frame = self.frame_processor(frame, self.shape[:-1])

        # Shuffle recent frames one step backward in time,
        # most recent frame at index -1.
        for i in xrange(self.shape[-1]-1):
            self.stacked[:,:,i] = self.stacked[:,:,i+1]
        self.stacked[:,:,-1] = frame

        return self.stacked
