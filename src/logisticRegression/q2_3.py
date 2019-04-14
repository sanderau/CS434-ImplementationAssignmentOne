import numpy as np
import csv
import math
#import matplotlib.pyplot as plt
import sys

#collects data from file and will put it into a numpy matrix
def data_collect(path):
	temp = []
	with open(path) as p:
		read = csv.reader(p)
		for r in read:
			temp.append(r)
	return np.array(temp).astype(float)

def sigmoid(wt, ti):
	return (1 / (1 + np.exp(-np.dot(wt, ti))))

#calculates the accuracy of this iteration
def accuracy(w, t, t_ans):
	
	accuracy = 0

	for i in range(len(t)):
		if t_ans[i] == 1:
			if sigmoid(np.transpose(w), t[i]) >= 1/2:
				accuracy += 1
		else:
			if (1-sigmoid(np.transpose(w), t[i])) >= 1/2:
				accuracy += 1
	return accuracy/len(t)


# main
def main():
	#see if we have all the command line arguments
	try:
		train_path = sys.argv[1]
		test_path = sys.argv[2]
		lambdas = []
		for i in range(3,len(sys.argv)):
			lambdas.append(sys.argv[i])
	except Exception as e:
		print("Did not have the correct command line arguments: " + str(e))
		sys.exit(1)

	print(lambdas)

	#collect the test and train data into matricies
	log_train = data_collect(train_path)
	log_test = data_collect(test_path)

	#get the training data and answers
	train = np.delete(log_train, 256, 1)
	train = train/256
	train = np.insert(train, 0, 1, axis=1)

	train_ans = np.delete(log_train, np.s_[0:256], 1)
	
	#get the testing data and answers
	test = np.delete(log_test, 256, 1)
	test = test/256
	test = np.insert(test, 0, 1, axis=1)

	test_ans = np.delete(log_test, np.s_[0:256], 1)
	
	#variables for the gradient descent
	gradient = True
	batches = 150 #number of batches
	j = 0
	w = np.zeros(train.shape[1]) # weight vector

	acc = [[], [], []]

	for l in range(0, len(lambdas)):

		#gradient descent
		while gradient:
			gradient_descent = np.zeros(train.shape[1])

			for i in range(train.shape[0]):
				prediction =  1 / (1 + math.exp(-np.dot(np.transpose(w), train[i])))
				gradient_descent += ((prediction - train_ans[i]) * train[i])
			
			w -= (.0001*(gradient_descent + (float(lambdas[l])*w)))
			j += 1

			acc[0].append(j)
			acc[1].append(accuracy(w, train, train_ans))
			acc[2].append(accuracy(w, test, test_ans))

			if j == batches:
				gradient = False
		filename = "lambda_" + str(lambdas[l]) + ".csv"
		np.savetxt(filename, acc, delimiter=",")

#	figure, axis = plt.subplots()

#	ax.plot(ll_ot[0], ll_ot[1], label="Training set")
#	ax.plot(ll_ot[0], ll_ot[1], label="Testing set")
#	ax.legend(loc="lower right")
#	plt.show

main()
