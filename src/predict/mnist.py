#!/usr/bin/env python3
import numpy as np
import struct as st
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


class NumberPredictor():
    def __init__(self, train_size, k):
        self.TRAIN_SIZE = train_size 
        self.k = k

        #X_train
        intType = np.dtype('int32').newbyteorder('>')
        nMetaBytes = 4 * intType.itemsize
        X_train = np.fromfile("data/train-images-idx3-ubyte", dtype="ubyte")
        magic_number, size, rows, cols = np.frombuffer(X_train[:nMetaBytes].tobytes(), intType)
        X_train = X_train[nMetaBytes:].astype(dtype='float32').reshape([size,rows*cols])
        X_train = X_train/255

        #Y_train
        Y_train = np.fromfile("data/train-labels-idx1-ubyte", dtype="ubyte")[2 * intType.itemsize:]

        #X_test
        X_test = np.fromfile("data/t10k-images-idx3-ubyte", dtype="ubyte")
        magic_number, size, rows, cols = np.frombuffer(X_test[:nMetaBytes].tobytes(), intType)
        X_test = X_test[nMetaBytes:].astype(dtype='float32').reshape([size,rows*cols])
        X_test = X_test/255

        #Y_test
        Y_test = np.fromfile("data/t10k-labels-idx1-ubyte", dtype="ubyte")[2 * intType.itemsize:]

        self.model = KNeighborsClassifier(n_neighbors=self.k)
        print("fitting model")
        self.model.fit(X_train[:self.TRAIN_SIZE], Y_train[:self.TRAIN_SIZE])
        print("finished fitting")

    def predict(self, img):
        return self.model.predict(img.reshape(1, -1))
        


if __name__ == "__main__":
    TRAIN_SIZE = 60000
    #X_train
    intType = np.dtype('int32').newbyteorder('>')
    nMetaBytes = 4 * intType.itemsize
    X_train = np.fromfile("../data/train-images-idx3-ubyte", dtype="ubyte")
    magic_number, size, rows, cols = np.frombuffer(X_train[:nMetaBytes].tobytes(), intType)
    X_train = X_train[nMetaBytes:].astype(dtype='float32').reshape([size,rows*cols])
    X_train = X_train/255

    #test = X_train[50] 
    #test = test.reshape([28,28])
    #plt.imshow(test, cmap="gray_r")
    #plt.show()


    #Y_train
    Y_train = np.fromfile("../data/train-labels-idx1-ubyte", dtype="ubyte")[2 * intType.itemsize:]

    #X_test
    X_test = np.fromfile("../data/t10k-images-idx3-ubyte", dtype="ubyte")
    magic_number, size, rows, cols = np.frombuffer(X_test[:nMetaBytes].tobytes(), intType)
    X_test = X_test[nMetaBytes:].astype(dtype='float32').reshape([size,rows*cols])
    X_test = X_test/255

    #Y_test
    Y_test = np.fromfile("../data/t10k-labels-idx1-ubyte", dtype="ubyte")[2 * intType.itemsize:]

    model = KNeighborsClassifier(n_neighbors=5)
    print("fitting model")
    model.fit(X_train[:TRAIN_SIZE], Y_train[:TRAIN_SIZE])
    print("finished fitting")


    X_test = X_test[:TRAIN_SIZE]
    predictions = model.predict(X_test)

    print(classification_report(Y_test[:TRAIN_SIZE], predictions))
