import sys
import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# if(len(sys.argv) < 3):
#     print("Must include filenames as arguments")
#
# else:
    # print(np.random.uniform(0,100,(433,1)))

    # training data
    # X = np.loadtxt(sys.argv[1], usecols=range(0,13)) #features
    # Y = np.loadtxt(sys.argv[1], usecols=range(13,14)) #desired outputs
    # Y = np.matrix(Y)
    # Y = Y.transpose()
    #
    # # testing data
    # Xtest = np.loadtxt(sys.argv[2], usecols=range(0,13)) #features
    # Ytest = np.loadtxt(sys.argv[2], usecols=range(13,14)) #desired outputs
    # Ytest = np.matrix(Ytest)
    # Ytest = Ytest.transpose()
    #
    # ASEtrains = []
    # ASEtests = []

X = np.genfromtxt("usps-4-9-train.csv", delimiter=',', usecols=range(256))
Y = np.genfromtxt("usps-4-9-train.csv", delimiter=',', usecols=range(256,257))
Y = np.matrix(Y).transpose()
print(X.shape)
print(X)
# print(Y.shape)
# print(Y.transpose().shape)
print()
# print(np.matrix(X).shape)
# print(np.matrix(X).transpose())
# print(np.matrix(X).transpose()[:,254])
