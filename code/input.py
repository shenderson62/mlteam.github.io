import os
import pandas as pd
import shutil


def divide_categories(ham10000Path):
    imagesDF = pd.read_csv(ham10000Path + '/HAM10000_metadata.csv')

    imagesDF.set_index('image_id', inplace = True)
    imagesNames = imagesDF.index

    # base processed images path
    processed = os.path.join(ham10000Path, 'processed')
    os.mkdir(processed)

    # paths for image categories in both datasets
    categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for category in categories:
        os.mkdir(os.path.join(processed, category))

    partOneFolder = os.path.join(ham10000Path, 'ham10000_images_part_1')
    partOneImages = os.listdir(partOneFolder)
    partTwoFolder = os.path.join(ham10000Path, 'ham10000_images_part_2')
    partTwoImages = os.listdir(partTwoFolder)

    for image in imagesNames:
        imageFileName = image + '.jpg'
        label = imagesDF.loc[image, 'dx']

        if imageFileName in partOneImages:
            src = os.path.join(os.path.join(partOneFolder, imageFileName))
            dst = os.path.join(processed, label, imageFileName)
            shutil.copyfile(src, dst)

        elif imageFileName in partTwoImages:
            src = os.path.join(os.path.join(partTwoFolder, imageFileName))
            dst = os.path.join(processed, label, imageFileName)
            shutil.copyfile(src, dst)

ham10000Path = 'D:/GeorgiaTech/3rdYear/1stSem/cs4641/HAM10000'
divide_categories(ham10000Path)