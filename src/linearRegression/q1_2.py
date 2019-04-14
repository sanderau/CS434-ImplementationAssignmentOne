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
