from keras.models import Sequential
from keras.layers import Conv2D

def convolutional(num_filters = 8, filter_size = 3):
    conLayer = Conv2D(num_filters,
                    filter_size,
                    input_shape=(28, 28, 3),
                    strides=2,
                    padding='same',
                    activation='relu'
                    )
    return conLayer