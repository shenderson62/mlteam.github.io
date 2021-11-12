import os
import pandas as pd
from tensorflow import keras
import tensorflow
from tensorflow.keras.preprocessing import image_dataset_from_directory
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def divide_categories(ham10000Path):
    imageDF = pd.read_csv(ham10000Path + '/HAM10000_metadata.csv')

    # divide image data into training and testing
    trainingDF, testingDF = train_test_split(imageDF, test_size=0.1, stratify = imageDF['dx'])
    trainingDF.set_index('image_id', inplace = True)
    testingDF.set_index('image_id', inplace = True)
    trainImages = trainingDF.index
    testImages = testingDF.index

    # base processed images path
    processed = os.path.join(ham10000Path, 'processed')
    os.mkdir(processed)

    # training images path
    training = os.path.join(processed, 'training')
    os.mkdir(training)

    # testing images path
    testing = os.path.join(processed, 'testing')
    os.mkdir(testing)

    # paths for image categories in both datasets
    categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for category in categories:
        os.mkdir(os.path.join(training, category))
        os.mkdir(os.path.join(testing, category))

    partOneFolder = os.path.join(ham10000Path, 'ham10000_images_part_1')
    partOneImages = os.listdir(partOneFolder)
    partTwoFolder = os.path.join(ham10000Path, 'ham10000_images_part_2')
    partTwoImages = os.listdir(partTwoFolder)

    for image in trainImages:
        imageFileName = image + '.jpg'
        label = trainingDF.loc[image, 'dx']

        if imageFileName in partOneImages:
            src = os.path.join(os.path.join(partOneFolder, imageFileName))
            dst = os.path.join(training, label, imageFileName)
            shutil.copyfile(src, dst)

        elif imageFileName in partTwoImages:
            src = os.path.join(os.path.join(partTwoFolder, imageFileName))
            dst = os.path.join(training, label, imageFileName)
            shutil.copyfile(src, dst)

    for image in testImages:
        imageFileName = image + '.jpg'
        label = testingDF.loc[image, 'dx']

        if imageFileName in partOneImages:
            src = os.path.join(os.path.join(partOneFolder, imageFileName))
            dst = os.path.join(testing, label, imageFileName)
            shutil.copyfile(src, dst)

        elif imageFileName in partTwoImages:
            src = os.path.join(os.path.join(partTwoFolder, imageFileName))
            dst = os.path.join(testing, label, imageFileName)
            shutil.copyfile(src, dst)

def generate_datasets(ham10000Path):
    categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    for category in categories:
        augment = os.path.join(ham10000Path, 'augment')
        os.mkdir(augment)
        
        img_dir = os.path.join(augment, 'img_dir')
        os.mkdir(img_dir)

        processed = os.path.join(ham10000Path, 'processed')
        training = os.path.join(processed, 'training')

        trainingCategory = os.path.join(training, category)
        trainingImgs = os.listdir(trainingCategory)

        for imgFile in trainingImgs:

            source = os.path.join(trainingCategory, imgFile)

            target = os.path.join(img_dir, imgFile)

            shutil.copyfile(source, target)

        # generator for augment images
        gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            zoom_range = 0.1,
            horizontal_flip = True,
            vertical_flip = True
        )

        augGen = gen.flow_from_directory(
            augment,
            save_to_dir = trainingCategory,
            save_format = 'jpg',
            target_size = (600, 450),
            batch_size = 100
        )

        imageNum = 5000 

        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((imageNum - num_files) / 100))

        for i in range(0, num_batches):
            images, labels = next(augGen)

        shutil.rmtree(augment)

