import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D

class Pooling(object):
    
    def init(self):
        pass

#image = image.reshape(1, 4, 4, 1)
    def perform_pooling(image):

 
        # define model containing just a single max pooling layer
        model = Sequential([MaxPooling2D(pool_size = 2, strides = 2)])
 
        # generate pooled output
        output = model.predict(image)
        return output
 
# # print output image
# output = np.squeeze(output)
# print(output)