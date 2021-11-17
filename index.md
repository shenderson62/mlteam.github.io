
# Project Midterm Report

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

### Data Cleaning
Where we received our data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
The images in HAM10000 have already been screened, filtered, and corrected so that the images present are all very high quality and clearly depict the skin lesions in question. Furthermore they are all clearly labeled with the distinct classifications they belong to with no ambiguous classifications present (Tschandl et al., 2018). We split our data into training and testing (validation) sets. 10% of our data is reserved for testing and 90% is for training.

### Convolutional Neural Network
To create a predictor using this data, we will create and train a Convolutional Neural Network to classify the skin lesions. CNNs are a class of neural networks with features such as shared weights or pooling that make it extremely suitable for analyzing visual data, which it is often used for (Albawi et al., 2017). The CNN will be trained on the entire HAM10000 dataset, using only the images and their categorizations as inputs. It will be tested on its performance of sorting pictures of skin lesions into their correct classifications.

## Results
The resulting predictor will show the effectiveness of using CNNs for this classification task. We saught to improve this effectiveness by changing the network's architecture through experimenting with the number and order of layers to find the setup that results in the most accurate CNN for classifying these skin lesions.

## Discussion
Although this visual classification task could be difficult since the pictures of skin lesions are naturally fine-grained, CNNs are very well suited for visual analysis so we don't expect to encounter too much difficulty in creating a relatively accurate predictor. It is possible for certain classifications of skin lesions to be more easily identified, which could limit accuracy. This would be handled by changing the architecture or manipulating specific hyperparameters. Additionally, CNNs can run into overfitting if the dataset is not large enough. However, since this dataset is one of the largest of its type by far, it is unlikely that this will be a problem (Tschandl et al., 2018).

One thing CNNs have a hard time accounting for is different locations of a feature in an image because it looks to match one precise shape to another precise shape. To account for this, we included a max pooling layer. This pooling function downsamples our images, creating a lower resolution of our images, and therefore retains the large structural elements of our feature without getting hindered by the fine details. Our max pooling function specifically classifies where the most prominent feature in the image is, which helps us account for small variations in feature location. We did this by creating a 2x2 filter with a stride of 2 that would reduce each row and column by half, therefore reducing resolution, while still retaining the major features of the image. 

When configuring our model, we chose categorical cross entropy as our loss function, since we have more than two categories within our classification problem. Categorical cross entropy uses one-hot encoding so that only the element within a vector corresponding to the correct class is 1, where the remaining elements would be zero (Biswas, 2021). We used Adam as our optimizer, which is a type of stochastic gradient descent, with a learning rate of .0005. Using a loss function helped us understand the difference between the predicted and actual probability distributions with regards to how our model classified the images of skin lesions.

To measure our predictive model performance, we used accuracy as one of our metrics.

![LossVisual](/assets/LossVisual.PNG)

As we see in the graph, our training and validation loss both decrease to a stable value, demonstrating a successful learning curve (Brownlee, 2019).

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
