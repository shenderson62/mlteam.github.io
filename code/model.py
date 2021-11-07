import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from convolutional import conLayer
from pooling import Pooling
from LossFunctionImplementation import lossAndOutput

model = Sequential([
    conLayer(),
    Pooling().perform_pooling,
    Flatten(),
    Dense(10, activation='softmax'),
])

model = lossAndOutput(model)
