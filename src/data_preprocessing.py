import os
import random
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from .preprocessing import *


def preprocessing(lPath1, lPath2):
    """Preprocess image before giving into model

    Args:
        lPath1 (str): directory of first dataset
        lPath2 (str): directory of second dataset

    Returns:
        tuple: Image data for CNN
    """    

    def process(imageList):
        """Generate training data arrays

        Args:
            imageList (list): list of paths of image

        Returns:
            tuple: two arrays used for preprocessing
        """        
        x, y = [], []

        for image in imageList:
            x.append(Preprocessor(image[0]).filter())
            print("\rProcessing Image " + image[0], end="")
            y.append(image[1])

        return x, y

    if not (os.path.exists(lPath1) and os.path.exists(lPath2)):
        raise Exception
    
    train_1 = [[os.path.join(lPath1, i), 0] for i in os.listdir(lPath1)]
    train_2 = [[os.path.join(lPath2, i), 1] for i in os.listdir(lPath2)]

    train_images = train_1 + train_2
    random.shuffle(train_images)

    del train_1
    del train_2
    gc.collect()

    X, y = process(train_images)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

    X_train  = np.array(X_train)
    X_val  = np.array(X_val)
    y_train  = np.array(y_train)
    y_val  = np.array(y_val)

    del X
    del y
    gc.collect()

    return X_train, X_val, y_train, y_val

