
# Project Proposal

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
The images in HAM10000 have already been screened, filtered, and corrected so that the images present are all very high quality and clearly depict the skin lesions in question. Furthermore they are all clearly labeled with the distinct classifications they belong to with no ambiguous classifications present (Tschandl et al., 2018). Thus we did not need to do anything further and can use the data as is.

### Convolutional Neural Network
To create a predictor using this data, we will create and train a Convolutional Neural Network to classify the skin lesions. CNNs are a class of neural networks with features such as shared weights or pooling that make it extremely suitable for analyzing visual data, which it is often used for (Albawi et al., 2017). The CNN will be trained on the entire HAM10000 dataset, using only the images and their categorizations as inputs. It will be tested on its performance of sorting pictures of skin lesions into their correct classifications.

## Potential Results
The resulting predictor will show the effectiveness of using CNNs for this classification task. We will seek to improve this effectiveness by changing the network's architecture through experimenting with the number and order of layers to find the setup that results in the most accurate CNN for classifying these skin lesions.

## Discussion
Although this visual classification task could be difficult since the pictures of skin lesions are naturally fine-grained, CNNs are very well suited for visual analysis so we don't expect to encounter too much difficulty in creating a relatively accurate predictor. It is possible for certain classifications of skin lesions to be more easily identified, which could limit accuracy. This would be handled by changing the architecture or manipulating specific hyperparameters. Additionally, CNNs can run into overfitting if the dataset is not large enough. However, since this dataset is one of the largest of its type by far, it is unlikely that this will be a problem (Tschandl et al., 2018).

## References
* Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
* Tschandl, Philipp, 2018, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", https://doi.org/10.7910/DVN/DBW86T, Harvard Dataverse, V3, UNF:6:/APKSsDGVDhwPBWzsStU5A== [fileUNF]
* S. Albawi, T. A. Mohammed and S. Al-Zawi, "Understanding of a convolutional neural network," 2017 International Conference on Engineering and Technology (ICET), 2017, pp. 1-6, doi: 10.1109/ICEngTechnol.2017.8308186.
* Whited, J. D., & Grichnik, J. M. (1998, March 4). Does This Patient Have a Mole or a Melanoma? Retrieved from https://jamanetwork.com/journals/jama/fullarticle/187305. 
* Convolutional Neural Network (CNN) &nbsp;: &nbsp; Tensorflow Core. TensorFlow. (n.d.). Retrieved November 15, 2021, from https://www.tensorflow.org/tutorials/images/cnn. 

## Timeline
![timeline](/assets/timeline.png)

## Responsibilities
![responsibilities](/assets/responsibilities.PNG)
