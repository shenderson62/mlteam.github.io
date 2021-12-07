
# Project Final Report

## Introduction/Background
Skin cancer is becoming exponentially prevalent in the United States. Due to the increasing risk of skin cancer, it is important to be able to diagnose malignant cancers before it reaches stages that are difficult to treat. Often times, skin cancer can be diagnosed looking at various traits of the skin lesion, such as the ABC’s (Whited & Grichnik, 2019):  

* Asymmetry
* Border
* Color

## Problem Definition
To help counterract this growth in skin cancer rates, we aim to use machine learning to evaluate images of potential skin cancer photos for such identifying traits and classify the images into what types of skin lesions. Due to the wide availability of cameras, such a predictor would help lighten the burden on the doctors currently doing the evaluations or even provide a second opinion in addition to theirs.

## Methods

### Data Collection
The skin cancer data used consists of the HAM10000 (“Human Against Machine with 10000 training images”) dataset, a collection of 10,015 dermatoscopic images categorized into seven different skin lesion classifications (Tschandl, 2018).

### Data Cleaning and Preprocessing
Where we received our data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T.

The images in HAM10000 have already been screened, filtered, and corrected so that the images present are all very high quality and clearly depict the skin lesions in question. Furthermore they are all clearly labeled with the distinct classifications they belong to with no ambiguous classifications present (Tschandl et al., 2018). 

Firstly, we organized the over ten thousand images into folders of their skin lesion categories through the metadata files provided with the dataset.

One thing we noticed was that the data was unbalanced. Although we are given plenty of images to learn with, their distribution between the categories of skin lesions is quite unbalanced, with the melanocytic nevi photos making up around 67% of the entire dataset. We considered many ways of handling this unbalance, such as augmenting the images, which we decided against as many of the transformations would be meaningless as the skin lesions, being circular, are reflectionally and rotationally symmetric, and ultimating decided upon giving the categories wieghts for the loss function. We weighted the categories so that the less images they contained, the greater the weight. In this way, the model gives greater value to underrepresented categories, as to make up for the bias against them stemming from their comparatively lesser amount of data.

We split our dataset into training and validation sets. 10% of our data is reserved for validation and 90% is for training. The seed used to randomly split the data was the same to guarantee that the training and validation sets are independent so the model's performance can be properly measured later on.

### Convolutional Neural Network
To create a predictor using this data, we created and trained a Convolutional Neural Network to classify the skin lesions. CNNs are a class of neural networks with features such as shared weights or pooling that make it extremely suitable for analyzing visual data, which it is often used for (Albawi et al., 2017). The CNN was trained on the entire HAM10000 dataset, using only the images and their categorizations as inputs. It was then tested on its performance of sorting pictures of skin lesions into their correct classifications.

The most important layers of a CNN are the convolutional layers which convolves a kernal with the image to produce a feature map, which works to denote the presence of various features within the image. We used a 3x3 kernal and sought 16 filters in our model.

The convolution operation may impose a certain degree of linearity onto an image. However, images may contain many non-linear aspects, such as gradient transitions between colors, borders, transitions between pixels, etc. In order to account for this, we apply a rectifier function to the featuere map to further increase the non-linearity of our images. In our model, we chose to use a Rectified Linear Unit (RELU) activation function that maintains only postive values in the image. After rectification, one could examine an image and notice that colors change more abruptly, indicating some linearity has been disposed of. 

One thing CNNs have a hard time accounting for is different locations of a feature in an image because it looks to match one precise shape to another precise shape. To account for this, we included a max pooling layer. This pooling function downsamples our images, creating a lower resolution of our images, and therefore retains the large structural elements of our feature without getting hindered by the fine details. Our max pooling function specifically classifies where the most prominent feature in the image is, which helps us account for small variations in feature location. We did this by creating a 2x2 filter with a stride of 2 that would reduce each row and column by half, therefore reducing resolution, while still retaining the major features of the image. 

When configuring our model, we chose categorical cross entropy as our loss function, since we have multiple categories within our classification problem. Categorical cross entropy uses one-hot encoding so that only the element within a vector corresponding to the correct class is 1, where the remaining elements would be zero (Biswas, 2021). This loss function was also modified through the weighing of the categories discussed earlier, to ensure that the categories with less data were still represented in their influence upon the model's learning.

Lastly the model is finished off with a fully connected layer, a layer of neurons that connect to every neuron in the previous flattened layer, which is just the data resulting from the convolutional and pooling layers converted into a one-dimensional vector, and use the softmax activation to predict which category of skin lesions an image belongs to.

We used Adam as our optimizer, which is a type of stochastic gradient descent, with a learning rate of .0001. Using a loss function helped us understand the difference between the predicted and actual probability distributions with regards to how our model classified the images of skin lesions.

## Results and Discussion
To measure our predictive model performance, we used its accuracy and loss as our metrics.

![LossVisual](/assets/LossVisual.PNG)

These graphs shows the model's accuracy and loss over 100 training epochs. 
In the accuracy graph, you can see that the validation accuracy ceases to grow meaningfully after around the 30th epoch, whereas the training accuracy continues to grow all the way to the 100th epoch. This suggests that the model is overfitting and corresponding too closely to the training data and 30 epochs is all we need for it to converge its maximum accuracy in classifying skin lesions, which is around 65% or so.
In the loss graph, the training loss occurs as expected, exponentially decreasing in value with each epoch. On the other hand, although the loss function initally decreases as well like expected, it then begins to increase linearly, even passing the training loss at one point. Similarly, this also suggests that the model is overfitting the training data (Brownlee, 2019).

Thus while the model does show some effectiveness at predicting the skin lesion category given an image, being accurate around 65% of the time, which is clearly better than the the random chance of around 14%, there is still room for improvement. In addition, the model shows signs of overfitting, which we could also work to prevent.

To handle overfitting, we can increase the amount of dropout applied or add more layers to decrease the interdependent learning between neurons. We can also stop the model early, which given our graphs is clearly around the 20-30 epoch, where the overfitting begins to occur.

To increase the accuracy of our CNN there are many changes we can try. For one, we could increase the depth of our model so that they can recognize more complex structures within the images. We could also use implement cross-validation, so that the model will eventually be trained over the entire dataset rather that a divided section of it. Furthermore, we could just simply experiment with modifying the model's hyperparameters such as epochs, batch size, learning rate, and numerous other variables.

For the final report, we will seek to test and implement these modifications to our model to increase its performance and fitting so that the model can be improved to a point where it is actually useful for the task of categorizing skin lesions.

## References
* Biswas, P. (2021, June 30). Importance of Loss Functions in Deep Learning and Python Implementation. Medium. Retrieved November 16, 2021, from https://towardsdatascience.com/importance-of-loss-functions-in-deep-learning-and-python-implementation-4307bfa92810. 
* Brownlee, J. (2019, August 6). How to use learning curves to diagnose machine learning model performance. Machine Learning Mastery. Retrieved November 17, 2021, from https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/. 
* Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
* Tschandl, Philipp, 2018, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", https://doi.org/10.7910/DVN/DBW86T, Harvard Dataverse, V3, UNF:6:/APKSsDGVDhwPBWzsStU5A== [fileUNF]
* S. Albawi, T. A. Mohammed and S. Al-Zawi, "Understanding of a convolutional neural network," 2017 International Conference on Engineering and Technology (ICET), 2017, pp. 1-6, doi: 10.1109/ICEngTechnol.2017.8308186.
* Whited, J. D., & Grichnik, J. M. (1998, March 4). Does This Patient Have a Mole or a Melanoma? Retrieved from https://jamanetwork.com/journals/jama/fullarticle/187305. 
* Convolutional Neural Network (CNN) &nbsp;: &nbsp; Tensorflow Core. TensorFlow. (n.d.). Retrieved November 15, 2021, from https://www.tensorflow.org/tutorials/images/cnn. 

## Timeline
![timeline](/assets/timeline.png)

## Responsibilities
![responsibilities](/assets/responsibilities.PNG)
