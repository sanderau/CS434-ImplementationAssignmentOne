import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, array, ones, linalg, zeros
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from collections import Counter

# myFuncs
def train(xTrain, yTrain):
    return


def predict(xTrain, yTrain, xTest, K):
    distances = []
    neighborsTargets = []

    for i in range( len( xTrain ) ):
        distance = np.sqrt( np.sum( np.square( xTest - xTrain[i, :] ) ) )
        distances.append([distance, i])

    distances = sorted(distances)

    for i in range(K):
        ind = distances[i][1]
        neighborsTargets.append(yTrain[ind][0])

    return Counter(neighborsTargets).most_common(1)[0][0]


def knn(xTrain, yTrain, xTest, preds, K):
    train(xTrain, yTrain)

    for i in range(len(xTest)):
        preds.append( predict(xTrain, yTrain, xTest[i, :], K) )

    return


def looCV(xTrain, yTrain, K):
    accuracy = 0

    for i in range(len(xTrain)):
        preds = []

        preds.append(predict(np.delete(xTrain, i, 0), np.delete(yTrain, i, 0), xTrain[i, :], K))
        # print(preds)
        preds = np.asarray(preds)
        # print(preds)
        accuracy += accuracy_score(yTrain[i], preds)
    return((1 - accuracy / 284)*100)


########################## MAIN ###############################
if(len(sys.argv) < 3):
    print("Must include filenames as arguments")
else:
    # trainDATA
    f = open(sys.argv[1], 'r')
    X = []
    Y = []
    f = csv.reader(f)
    for row in f:
        X.append(row[1:31])
        Y.append(row[0:1])

    xTrain = array(X).astype(np.float64)
    # xTrain = normalize(xTrain, axis=1, norm='l1')
    xTrain = normalize(xTrain, axis=0, norm='max')
    yTrain = array(Y).astype(np.float64)

    # testDATA
    xTest = []
    yTest = []

    f = open(sys.argv[2], 'r')
    f = csv.reader(f)
    for row in f:
        yTest.append(row[0:1])
        xTest.append(row[1:31])

    xTest = array(xTest).astype(np.float64)
    # xTest = normalize(xTest, axis=1, norm='l1')
    xTest = normalize(xTest, axis=0, norm='max')
    yTest = array(yTest).astype(np.float64)

    trainE = []
    cvE = []
    testE = []

    oddK = list(range(1,52))[0::2]
    for i in range(len(oddK)):
        preds = []

        knn(xTrain, yTrain, xTrain, preds, oddK[i])
        preds = np.asarray(preds)
        accuracy = (1.0 - accuracy_score(yTrain, preds))*100
        print("\nK: ", oddK[i],   "\tTrain error: \t", accuracy)
        trainE.append(accuracy)

        accuracy = looCV(xTrain, yTrain, oddK[i])
        print("K: ", oddK[i],   "\tlooCV error: \t", accuracy)
        cvE.append(accuracy)

        preds = []
        knn(xTrain, yTrain, xTest, preds, oddK[i])
        preds = np.asarray(preds)
        accuracy = (1.0 - accuracy_score(yTest, preds))*100
        print("K: ", oddK[i],   "\tTest error: \t", accuracy)
        testE.append(accuracy)

    ########################## PLOT ###############################
    plt.plot(oddK, trainE)
    plt.plot(oddK, cvE)
    plt.plot(oddK, testE)

    plt.xlabel('K values')
    plt.ylabel('Percent Error')
    plt.title('Percent Error vs K')
    plt.legend(['Train Error', 'cv Error', 'Test Error'], loc='lower right')
    plt.show()
