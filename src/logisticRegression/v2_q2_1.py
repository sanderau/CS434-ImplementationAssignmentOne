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
def get_acc(w, t, t_ans):
	
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
		learning_rate = sys.argv[3]
	except:
		print("Did not have the correct command line arguments")
		sys.exit(1)

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
        lambda_ = [.01, .1, 1, 10, 100, 1000]

	ll_ot = [[], [], []]

	#gradient descent
	while gradient:
		gradient_descent = np.zeros(train.shape[1])

		for i in range(train.shape[0]):
			prediction =  1 / (1 + math.exp(-np.dot(np.transpose(w), train[i])))
			gradient_descent += ((prediction - train_ans[i]) * train[i])
		
		w -= (.0001*(gradient_descent + (lambda_[0]*w)))
		j += 1

		ll_ot[0].append(j)
		ll_ot[1].append(get_acc(w, train, train_ans))
		ll_ot[2].append(get_acc(w, test, test_ans))

		if j == batches:
			gradient = False

	np.savetxt("q2.csv", ll_ot, delimiter=",")

#	figure, axis = plt.subplots()

#	ax.plot(ll_ot[0], ll_ot[1], label="Training set")
#	ax.plot(ll_ot[0], ll_ot[1], label="Testing set")
#	ax.legend(loc="lower right")
#	plt.show

main()
