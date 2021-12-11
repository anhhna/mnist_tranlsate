import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def ReadFile(train_file, test_file):
    dirpath = os.getcwd()
    train = pd.read_csv(dirpath + '/' + train_file)
    test = pd.read_csv(dirpath + '/' + test_file)

    X_test = test.drop(['label'], axis=1)
    y_test = test[['label']]

    X_train = train.drop(['label'], axis=1)
    y_train = train[['label']]

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = X_train.reshape(X_train.shape[0], 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 28, 28)

    return X_train, X_test


def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted


def random_translate(image):
    # define the translation amount as a fraction of the image width
    tx = np.random.randint(-5, 6)
    ty = np.random.randint(-5, 6)
    # compute the transformation matrix
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # apply the transformation
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


if __name__== '__main__':

    X_train, X_test = ReadFile('mnist_train.csv', 'mnist_test.csv')

    dirpath = os.getcwd()
    new_train = pd.read_csv(dirpath + '/mnist_train.csv')
    n = X_train.shape[0]
    # create new dataframe from train
    for i in range(n):
        # add translation to each image
        img = X_train[i]
        img = random_translate(img.astype(np.uint8))
        new_train.iloc[i, 1:] = img.reshape(784)
    new_train.to_csv(dirpath + '/mnist_train_translated.csv', index=False)

    new_test = pd.read_csv(dirpath + '/mnist_test.csv')
    n = X_test.shape[0]
    # create new dataframe from train
    for i in range(n):
        # add translation to each image
        img = X_test[i]
        img = random_translate(img.astype(np.uint8))
        new_test.iloc[i, 1:] = img.reshape(784)
    new_test.to_csv(dirpath + '/mnist_test_translated.csv', index=False)
