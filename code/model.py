import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.utils import to_categorical
from convolutional import convolutional
from pooling import Pooling
from LossFunctionImplementation import LossFunctionImplementation
import matplotlib.pyplot as plt

# input path to HAM10000 dataset
ham10000Path = ''
processed = os.path.join(ham10000Path, 'processed')
training = os.path.join(processed, 'training')
testing = os.path.join(processed, 'testing')

trainingDS = image_dataset_from_directory(
    training, 
    image_size = (600, 450), 
    batch_size = 250
)

testingDS = image_dataset_from_directory(
    testing,
    image_size = (600, 450),
    batch_size = 250
)

model = Sequential()
model.add(Conv2D(8,3,padding="same", activation="relu", input_shape=(600, 450, 3)))
model.add(Pooling().perform_pooling())

model.add(Conv2D(8,3,padding="same", activation="relu"))
model.add(Pooling().perform_pooling())

model.add(Conv2D(8,3,padding="same", activation="relu"))
model.add(Pooling().perform_pooling())

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(7,activation="relu"))

model = LossFunctionImplementation.lossAndOutput(model)

data = model.fit(
    trainingDS,
    epochs = 100,
    validation_data = testing
 )

trainAccuracy = data.history['accuracy']
testAccuracy = data.history['val_accuracy']
trainLoss = data.history['loss']
testLoss = data.history['val_loss']

epochRange = range(100)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochRange, trainAccuracy, label='Training Accuracy')
plt.plot(epochRange, testAccuracy, label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochRange, trainLoss, label='Training Loss')
plt.plot(epochRange, testLoss, label='Testing Loss')
plt.legend(loc='upper right')
plt.title('Training and Testing Loss')
plt.show()