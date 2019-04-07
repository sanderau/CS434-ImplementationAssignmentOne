import numpy as np

path = "../../docs/housing_train.txt"

def countLines():
	with open(path) as f:
		for i, l in enumerate(f):
			pass
	return i + 1;

def getData(X,Y, numLines):
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
	numLines = countLines()

	Y = np.zeros(shape=(numLines, 1))
	X = np.zeros(shape=(numLines, 13))

	X, Y = getData(X,Y, numLines)

	w = (np.dot(np.transpose(X), Y)) / (np.dot(np.transpose(X), X))

	print(w)

if __name__ == "__main__":
	main()
