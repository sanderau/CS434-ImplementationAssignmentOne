import sys
import numpy as np
from numpy import arange, array, ones, linalg, zeros
import csv
import math
import matplotlib.pyplot as plt
np.seterr(over='ignore')

accuracy = []
accuracyTest = []

if(len(sys.argv) < 4):
    print("Must include filenames and LearningRate as arguments")
else:
    # trainDATA
    f = open(sys.argv[1], 'r')
    X = []
    Y = []
    f = csv.reader(f)
    for row in f:
        X.append(row[0:256])
        Y.append(row[256:257])

    X = array(X)
    X = np.longdouble(X)
    Y = array(Y)
    Y = np.longdouble(Y)

    # testDATA
    f = open(sys.argv[2], 'r')
    Xtest = []
    Ytest = []
    f = csv.reader(f)
    for row in f:
        Xtest.append(row[0:256])
        Ytest.append(row[256:257])

    Xtest = array(Xtest)
    Xtest = np.longdouble(Xtest)
    Ytest = array(Ytest)
    Ytest = np.longdouble(Ytest)

    LR = float(sys.argv[3]) # reading Learning rate from command args

    def sigmoid(w, x): #sigmoid func
        wTx = np.dot(-w.T, np.matrix(x).T)
        dnmtr = np.longdouble(1 + np.exp(wTx))
        z = np.longdouble(1/dnmtr)
        return z

    # train
    def batchGD(X, Y, LR, its):
        W = zeros(256)
        while(its > 0):

            D = zeros(256)
            acc = 0
            for i in range(X.shape[0]):
                x = X[i] # one row of features of Xmatrix
                y = Y[i] # row Ymatrix

                yHat = sigmoid(W, x)
                if yHat.all() > .4:
                    yHat = 1
                else:
                    yHat = 0

                err = y - yHat
                D = D + (err * x)

                if(err == 0):
                    acc += 1 # correct guess add to accuracy count

            accuracy.append(acc/X.shape[0]) #accuracy append TRAIN
            W = W + (LR * D) # update
            its -= 1

            test(Xtest, Ytest, W)

        return W

    def lossLog(yValue, guessed): # log still a little confused on this implemented by looking at online resources
        epsilon = 1e-15
        P = np.clip(guessed, epsilon, (1 - epsilon))
        if yValue == 1:
            if P[0] == 0:
                P[0] = 10**-10
            return -np.log(P[0])
        else:
            if P[0] == 1:
                P[0] = 1 - (10**-10)
            return -np.log(1 - P[0])

    def test(Xtest, Ytest, W):
        Loss = 0
        acc = 0
        for i in range(Xtest.shape[0]):
            x = Xtest[i]
            y = Ytest[i]

            yHat = sigmoid(W, x) # guess
            Loss += np.longdouble(lossLog(y, yHat)) #loss append

            if(y - yHat == 0): # correct guess? add to accuracy count
                acc += 1
        accuracyTest.append(acc/Xtest.shape[0]) #get accuracy percent for testDATA
        return Loss

    wVector = batchGD(X, Y, LR, 145) #batch gradient descent iterations with TRAIN
    lossTotal = test(Xtest, Ytest, wVector) #run on TESTDATA


    print("W = ", wVector)
    print("\n\n")
    print("Loss = ", lossTotal)

    del accuracyTest[-1] #extra accuracy value for test, remove
    xRange = list(range(0,145))


    plt.xlabel('Gradient Descent Iterations')
    plt.ylabel('Training Accuracy')
    plt.title('Training Data ')
    plt.plot(xRange, accuracy, 'ro')
    plt.axis([0, 160, 0, 1])
    plt.show()

    plt.xlabel('Gradient Descent Iterations')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Data ')
    plt.plot(xRange, accuracyTest, 'ro')
    plt.axis([0, 150, 0, 1])
    plt.show()
