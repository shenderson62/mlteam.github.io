from math import log
import tensorflow_estimator as tf
from keras.models import Sequential
from keras.layers import Conv2D
class LossFunctionImplementation:
    """
    Actual implementation of loss_cross_entropy.
    Source where I learned how to implement: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    """
    def loss_cross_entropy(actual, predicted) :
        sum = 0.0
        for i in range(len(actual)) :
            sum += actual[i] * log(1e-16 + predicted[i])
        mean = (1.0 / (len(actual))) * sum
        return -mean

    """"
    Source where I learned how to use tensorflow to compute loss function & use sigmoid activation function.
    https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    """
    def lossAndOutput(model) :
        loss = 'binary_crossentropy'
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        return model