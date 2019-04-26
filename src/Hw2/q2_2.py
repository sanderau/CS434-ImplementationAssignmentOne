import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, array, ones, linalg, zeros
from collections import Counter

varG = 0
tE = 0

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


def testE(mtrxTEST, feat, thres, d, ltLBL, rtLBL):
    s1, s2 = [], []
    ltStat, rtStat = 0, 0
    ltError, ltError = 0, 0

    for j in range(len(mtrxTEST)):
        if(mtrxTEST[j][feat] >= thres): # True
            s1.append(mtrxTEST[j])
        else:                       # False
            s2.append(mtrxTEST[j])

    pos = np.count_nonzero( np.array(s1)[:, -1] == ltLBL)
    neg = np.count_nonzero( np.array(s1)[:, -1] == rtLBL)

    if (pos == 0 or neg == 0):
        ltStat = 1
    ltError = neg

    pos2 = np.count_nonzero( np.array(s2)[:, -1] == ltLBL)
    neg2 = np.count_nonzero( np.array(s2)[:, -1] == rtLBL)

    if (pos2 == 0 or neg2 == 0):
        rtStat = 1
    rtError = pos2

    for i in range(d):
        print(' | ', end='')
    print(" label", ltLBL, " >= ", thres, "\t\t\t\t[ ", pos, ", ", neg, "\t]")
    for i in range(d):
        print(' | ', end='')
    print(" label", rtLBL, " < ", thres, "\t\t\t\t[ ", pos2, ", ", neg2, "\t]")

    return s1, ltStat, s2, rtStat, ltError,  rtError


def trainE(mtrx, feat, thres, d):
    s1, s2 = [], []
    ltStat, rtStat = 0, 0
    ltError, ltError = 0, 0

    for j in range(len(mtrx)):
        if(mtrx[j][feat] >= thres): # True
            s1.append(mtrx[j])
        else:                       # False
            s2.append(mtrx[j])

    majLeft = Counter(np.array(s1)[:, -1]).most_common(1)[0][0]
    majRight = Counter(np.array(s2)[:, -1]).most_common(1)[0][0]
    # both sides have the same label, give it to side that has more correct
    if(majLeft == majRight):
        lt = Counter(np.array(s1)[:, -1]).most_common(1)[0][1]
        rt = Counter(np.array(s2)[:, -1]).most_common(1)[0][1]
        if lt >= rt:
            majRight = majRight * -1
        else:
            majLeft = majLeft * -1

    pos = np.count_nonzero( np.array(s1)[:, -1] == majLeft)
    neg = np.count_nonzero( np.array(s1)[:, -1] == majRight)

    if (pos == 0 or neg == 0):
        ltStat = 1
    ltError = neg

    pos2 = np.count_nonzero( np.array(s2)[:, -1] == majLeft)
    neg2 = np.count_nonzero( np.array(s2)[:, -1] == majRight)

    if (pos2 == 0 or neg2 == 0):
        rtStat = 1
    rtError = pos2

    return s1, ltStat, s2, rtStat, ltError, rtError, majLeft, majRight


def dTree(xTrain, d, max, E, Test, errorT):
    gain, thres, feat = 0, 0, None

    global varG
    global tE

    if(d >= max):
        varG += E/284
        tE += errorT/284
        return

    for j in range(len(xTrain[0,:]) - 1):
        g, t = iGain(xTrain, j)
        if(g > gain):
            gain = g
            thres = t
            feat = j
    s1, s1Stat, s2, s2Stat, ltE, rtE, ltLBL, rtLBL = trainE(xTrain, feat, thres, d)
    s1Test, s1Tstat, s2Test, s2Tstat, ltEtest,  rtEtest = testE(Test, feat, thres, d, ltLBL, rtLBL)

    if(s1Stat != 1 and s1Tstat != 1):
        dTree(np.array(s1), d+1, max, ltE, np.array(s1Test), ltEtest)
    else:
        tE += ltEtest/284
    if(s2Stat != 1 and s2Tstat != 1):
        dTree(np.array(s2), d+1, max, rtE, np.array(s2Test), rtEtest)
    else:
        tE += rtEtest/284
    return

########################## MAIN ###############################
if(len(sys.argv) < 4):
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
    trainError = []
    testError = []

    for i in range(int(sys.argv[3])):
        dTree(Train, 1, i+2, 0, Test, 0)
        print("Training Error %: ", varG * 100)
        trainError.append(varG * 100)
        print("Testing Error %: ", tE * 100)
        testError.append(tE * 100)

        varG = 0
        tE = 0

    print("\n\n")
    print("Train Error %: ", trainError)
    print("Test Error %: ", testError)
    ########################## PLOT ###############################
    plt.plot([1,2,3,4,5], trainError)
    # plt.plot(oddK, cvE)
    plt.plot([1,2,3,4,5], testError)

    plt.xlabel('d value')
    plt.ylabel('Percent Error')
    plt.title('Percent Error vs Depth')
    plt.legend(['Train Error', 'Test Error'], loc='upper right')
    plt.show()
