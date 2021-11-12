import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D

class Pooling(object):
    def init(self):
        pass

#image = image.reshape(1, 4, 4, 1)
    def perform_pooling(self):
        # define model containing just a single max pooling layer
        pool = MaxPooling2D(pool_size = 2, strides = 2)
        return pool