<<<<<<< HEAD
import sys
import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

if(len(sys.argv) < 3):
    print("Must include filenames as arguments")

else:
    # training data
    X = np.loadtxt(sys.argv[1], usecols=range(0,13)) #features
    Y = np.loadtxt(sys.argv[1], usecols=range(13,14)) #desired outputs
    Y = np.matrix(Y)
    Y = Y.transpose()

    dummyCol = np.ones((433,1)) #dummy variable column
    dummyX = np.append(dummyCol, X, axis = 1) #matrix with dummy column

    XT = dummyX.transpose()
    XTX = XT.dot(dummyX)
    XTXinv = inv(XTX)
    W = XTXinv.dot(XT.dot(Y))

    WT = W.transpose()
    WTxXT = WT.dot(XT)

    SE = np.square(Y - WTxXT.transpose())
    ASE = SE.sum() / 443


    # testing data
    Xtestraw = np.loadtxt(sys.argv[2], usecols=range(0,13)) #features
    Ytest = np.loadtxt(sys.argv[2], usecols=range(13,14)) #desired outputs
    Ytest = np.matrix(Ytest)
    Ytest = Ytest.transpose()

    dummyCol = np.ones((74,1)) #dummy variable column
    Xtest = np.append(dummyCol, Xtestraw, axis = 1) #matrix with dummy column

    XTtest = Xtest.transpose()
    XTXtest = XTtest.dot(Xtest)
    XTXinvTest = inv(XTXtest)

    WTxXTtest = WT.dot(XTtest)

    SEtest = np.square(Ytest - WTxXTtest.transpose())
    ASEtest = SEtest.sum() / 73

    print("W = \n", W)
    print("\n")

    print("ASE over the training data: ", ASE)
    print("ASE over the testing data: ", ASEtest)

# print(W.shape)
# print(dummyX[0].shape)
# X0 = np.matrix(dummyX[0])
# X432 = np.matrix(dummyX[432])
# print(X0.shape)
# Wt = W.transpose()
# print(W.dot(np.matrix(dummyX[0])))
# print(W.transpose().dot(X0.transpose()))
# print("\n\n\n")
# print(dummyX[432])
# print(W)
# print(W.transpose().dot(X432.transpose()))

# bos = pd.DataFrame(X)
# bos.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
#                 'PTRATIO', 'B', 'LSTAT']
# bos['PRICE'] = Y
#
# Y1 = bos['PRICE']
# X1 = bos.drop('PRICE', axis = 1)

# XT      = X.transpose()
# XTX     = XT.dot(X)
# XTXinv  = inv(XTX)
# W = XTXinv.dot(XT.dot(Y))


#test
# Xtest = np.matrix([
#                 [1, 50,     166],
#                 [1, 57,     196],
#                 [1, 50,     191],
#                 [1, 53.34,  180.34],
#                 [1, 54,     174],
#                 [1, 55.88,  176.53],
#                 [1, 57,     177],
#                 [1, 55.88,  208.28],
#                 [1, 57,     199],
#                 [1, 54,     181],
#                 [1, 55,     178],
#                 [1, 53,     172],
#                 [1, 57,     185],
#                 [1, 49.5,   165],
#                 [1, 57,     188]
#             ])
# Ytest = np.matrix([
#                 [170],
#                 [191],
#                 [189],
#                 [180.34],
#                 [171],
#                 [176.53],
#                 [187],
#                 [185.42],
#                 [190],
#                 [181],
#                 [180],
#                 [175],
#                 [188],
#                 [170],
#                 [185]
#             ])
# print(Ytest.transpose().shape)
=======
import numpy as np
import csv
import math
import sys

def countLines(path):
	with open(path) as f:
		for i, l in enumerate(f):
			pass
	return i + 1;

def getData(X,Y, numLines, path):
	Y = np.zeros(shape=(numLines, 1))
	X = np.zeros(shape=(numLines, 13))
	
	with open(path) as data:
		line = data.readline()
		cnt = 0

		while line:
			split = line.split()

			for i in range(0,12):
				X[cnt][i] = float(split[i])
				
			Y[cnt][0] = float(split[13])

			cnt = cnt + 1
			line = data.readline()
	return X, Y



def main():
	#test to see if the files are included in test params
	try:
		trainPath = sys.argv[1]
		testPath = sys.argv[2]
	except:
		print("Could not find files, or they were not included on the command line")
		sys.exit(1)
	#old code I might reuse
	numTrainLines = countLines(trainPath)
	numTestLines = countLines(testPath)
	
	#get information from file
	with open(trainPath, "r") as fp:
		matrix_train = np.array([line.split() for line in fp]).astype(float)
	
	with open(testPath, "r") as fp:
		matrix_test = np.array([line.split() for line in fp]).astype(float)

	#extract information into numpy matrix
	X_train = np.delete(matrix_train, 13, 1)
	Y_train = np.delete(matrix_train, np.s_[0:13], 1)

	#extract info for test
	X_test = np.delete(matrix_test, 13, 1)
	Y_test = np.delete(matrix_test, np.s_[0:13], 1)

#	X_tran, Y_train = getData(X_train, Y_train, numTrainLines, trainPath)

	#create dummy data
	X_train_dummy = np.insert(X_train, 0, 1, axis=1)
	X_test_dummy = np.insert(X_test, 0, 1, axis=1)

	#calculate left and right hand side equations for one w/o dummy data
	left = np.linalg.inv(np.dot(np.transpose(X_train), X_train))
	right = np.dot(np.transpose(X_train), Y_train)

	#calculate left and right for dummy data
	dummy_left = np.linalg.inv(np.dot(np.transpose(X_train_dummy), X_train_dummy))
	dummy_right = np.dot(np.transpose(X_train_dummy), Y_train)
	
	#calculate real weight
	w = np.dot(left, right)
	dummy_w = np.dot(dummy_left, dummy_right)


	print("Real weight: " + str(w))
	print("Dummy weight: " + str(dummy_w))


main()
>>>>>>> 35e8d7635105e6a1453c75587cbaaeafa6bd2403
