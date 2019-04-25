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
        pos = np.count_nonzero( mtrx[ :, -1] == 1)
        neg = np.count_nonzero( mtrx[ :, -1] == -1)
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
    # print()
    if(maj1 == 1):
        maj2 = -1
    else:
        maj2 = 1

    pos1 = np.count_nonzero( np.array(s1)[:, -1] == maj1)
    pos2 = np.count_nonzero( np.array(s2)[:, -1] == maj2)

    # tE = (1 - ((pos1+pos2)/284))*100

    return maj1, maj2, s1, s2


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
    ltLBL, rtLBL, s1, s2 = trainE(xTrain, feat, thres)

    return feat, thres, ltLBL, rtLBL, s1, s2

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
    feat, thres, ltLBL, rtLBL, lt, rt = dTree(Train)
    pos = np.count_nonzero( np.array(lt)[:, -1] == ltLBL)
    neg = np.count_nonzero( np.array(lt)[:, -1] == rtLBL)

    pos2 = np.count_nonzero( np.array(rt)[:, -1] == ltLBL)
    neg2 = np.count_nonzero( np.array(rt)[:, -1] == rtLBL)

    print("Decision Stump: ")
    print("Feature ", feat, " split")
    print("| Label ", ltLBL, " < ", thres, "[ ", pos, ", ", neg, " ]")
    print("| Label ", rtLBL, " >= ", thres, "[ ", pos2, ", ", neg2, " ]")

    print("\n\nSplitting on feature ", feat)
    print("At threshold value ", thres)

    
