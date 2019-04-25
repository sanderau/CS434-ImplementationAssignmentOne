import csv
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, array, ones, linalg, zeros
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from collections import Counter

def hS(mtrx):
    if(mtrx.size != 0 ):
        pos = np.count_nonzero( mtrx[ :, 30] == 1)
        neg = np.count_nonzero( mtrx[ :, 30] == -1)
        if(pos == 0 or neg == 0):
            return 0
        h = (-pos/(pos+neg)) * (math.log( pos/(pos+neg), 2))-(neg/(pos+neg))*(math.log(neg/(pos+neg),2))
        return h
    return 0


def pH(mtrx, parentSize):
    if(mtrx.size != 0 ):
        pos = np.count_nonzero( mtrx[ :, -1] == 1)
        neg = np.count_nonzero( mtrx[ :, -1] == -1)
        if(pos == 0 or neg == 0):
            return 0
        h = (-pos/(pos+neg)) * (math.log( pos/(pos+neg), 2))-(neg/(pos+neg))*(math.log(neg/(pos+neg),2))
        p = (pos+neg)/parentSize
        return (h*p)
    return 0


def iGain(mtrx, feature):
    sortCol = mtrx[mtrx[:,feature].argsort()][:,feature]
    h = hS(mtrx)

    bestGain = 0
    bestThres = 0

    for i in range(len(sortCol)):
        s1, s2 = [], []
        for j in range(len(mtrx)):
            if(mtrx[j][feature] >= sortCol[i]): # True
                s1.append(mtrx[j])
            else:                               # False
                s2.append(mtrx[j])
        pH1 = pH(np.array(s1), len(mtrx) )
        pH2 = pH(np.array(s2), len(mtrx) )

        iG = h - ( pH1 + pH2 )
        if iG > bestGain:
            bestThres = sortCol[i]
            bestGain = iG

    return bestGain, bestThres


def testE(mtrxTEST, feat, thres, ltLBL, rtLBL):
    s1, s2 = [], []

    for j in range(len(mtrxTEST)):
        if(mtrxTEST[j][feat] >= thres): # True
            s1.append(mtrxTEST[j])

        else:                       # False
            s2.append(mtrxTEST[j])

    pos1 = np.count_nonzero( np.array(s1)[:, -1] == ltLBL)
    pos2 = np.count_nonzero( np.array(s2)[:, -1] == rtLBL)

    print( "Testing Error %: ", (1 - ((pos1+pos2)/284))*100 )


def trainE(mtrx, feat, thres):
    s1, s2 = [], []

    for j in range(len(mtrx)):
        if(mtrx[j][feat] >= thres): # True
            s1.append(mtrx[j])

        else:                       # False
            s2.append(mtrx[j])

    maj1 = Counter(np.array(s1)[:, -1]).most_common(1)[0][0]
    maj2 = Counter(np.array(s2)[:, -1]).most_common(1)[0][0]

    pos1 = np.count_nonzero( np.array(s1)[:, -1] == maj1)
    pos2 = np.count_nonzero( np.array(s2)[:, -1] == maj2)

    tE = (1 - ((pos1+pos2)/284))*100

    return maj1, maj2, tE


def dTree(xTrain):
    gain = 0
    thres = 0
    feat = None

    for j in range(len(xTrain[0,:]) - 1):
        g, t = iGain(xTrain, j)
        if(g > gain):
            gain = g
            thres = t
            feat = j
    lt, rt, tE = trainE(xTrain, feat, thres)

    return feat, thres, lt, rt, tE, gain

########################## MAIN ###############################
if(len(sys.argv) < 3):
    print("Must include filenames as arguments")
else:
    # trainDATA
    X = []
    f = open(sys.argv[1], 'r')
    f = csv.reader(f)
    for row in f:
        X.append(row[1:31]+row[0:1])
    Train = array(X).astype(np.float64)

    # testDATA
    X = []
    f = open(sys.argv[2], 'r')
    f = csv.reader(f)
    for row in f:
        X.append(row[1:31] + row[0:1])
    Test = array(X).astype(np.float64)
    ########################## TEST ###############################
    feat, thres, lt, rt, trainE, grain = dTree(Train)

    print("Decision Stump: ")
    print("Feature ", feat, " split")
    print("| Label ", lt, " < ", thres)
    print("| Label ", rt, " >= ", thres)

    print("\n\nSplitting on feature ", feat)
    print("With threshold value ", thres)
    print("Best information gain value ", grain)

    print("\n\nTraining Error %: ", trainE)
    testE(Test, feat, thres, lt, rt)
    ########################## PLOT ###############################
    # plt.plot(oddK, trainE)
    # plt.plot(oddK, cvE)
    # plt.plot(oddK, testE)
    #
    # plt.xlabel('K values')
    # plt.ylabel('Percent Error')
    # plt.title('Percent Error vs K')
    # plt.legend(['Train Error', 'cv Error', 'Test Error'], loc='lower right')
    # plt.show()
