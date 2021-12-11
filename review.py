import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


if __name__== '__main__':

    X_train, X_test = ReadFile('mnist_train.csv', 'mnist_test.csv')
    # show x_train
    # plot n images
    n = 10
    plt.figure(figsize=(20, 20))
    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i * n + j + 1)
            plt.imshow(np.uint8(X_train[i * n + j]), cmap='gray')
            plt.axis('off')
    plt.show()

    plt.figure(figsize=(20, 20))
    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i * n + j + 1)
            plt.imshow(np.uint8(X_test[i * n + j]), cmap='gray')
            plt.axis('off')
    plt.show()

    X_train, X_test = ReadFile('mnist_train_translated.csv', 'mnist_test_translated.csv')
    # show x_train
    # plot n images
    plt.figure(figsize=(20, 20))
    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i * n + j + 1)
            plt.imshow(np.uint8(X_train[i * n + j]), cmap='gray')
            plt.axis('off')
    plt.show()

    plt.figure(figsize=(20, 20))
    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i * n + j + 1)
            plt.imshow(np.uint8(X_test[i * n + j]), cmap='gray')
            plt.axis('off')
    plt.show()
