import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

if(len(sys.argv) < 3):
    print("Must include filenames as arguments")

else:
    # print(np.random.uniform(0,100,(433,1)))

    # training data
    X = np.loadtxt(sys.argv[1], usecols=range(0,13)) #features
    Y = np.loadtxt(sys.argv[1], usecols=range(13,14)) #desired outputs
    Y = np.matrix(Y)
    Y = Y.transpose()

    # testing data
    Xtest = np.loadtxt(sys.argv[2], usecols=range(0,13)) #features
    Ytest = np.loadtxt(sys.argv[2], usecols=range(13,14)) #desired outputs
    Ytest = np.matrix(Ytest)
    Ytest = Ytest.transpose()

    ASEtrains = []
    ASEtests = []

    # for loop of d = 0...10 features
    for i in range(5):

            # dummyCol = np.ones((433,1)) #dummy variable column
            randFeat1 = np.random.uniform(0,100,(433,1)) #random feature
            randFeat2 = np.random.uniform(0,100,(433,1)) #random feature
            X = np.append(randFeat1, X, axis = 1) #matrix with rand1 column
            X = np.append(randFeat2, X, axis = 1) #matrix with rand2 column

            # training data ASE
            XT = X.transpose()
            XTX = XT.dot(X)
            XTXinv = inv(XTX)
            W = XTXinv.dot(XT.dot(Y))

            WT = W.transpose()
            WTxXT = WT.dot(XT)

            SE = np.square(Y - WTxXT.transpose())
            ASE = SE.sum() / 443


            # testing data
            randFeat1 = np.random.uniform(0,100,(74,1))
            randFeat2 = np.random.uniform(0,100,(74,1))
            Xtest = np.append(randFeat1, Xtest, axis = 1) #matrix with rand1 column
            Xtest = np.append(randFeat2, Xtest, axis = 1) #matrix with rand2 column

            # testing data ASE
            XTtest = Xtest.transpose()
            XTXtest = XTtest.dot(Xtest)
            XTXinvTest = inv(XTXtest)

            WTxXTtest = WT.dot(XTtest)

            SEtest = np.square(Ytest - WTxXTtest.transpose())
            ASEtest = SEtest.sum() / 73

            print("\n\n__________________________________")
            print("Linear regression with d = ", (i+1) * 2)
            print("W = \n", W)
            print("\n")

            print("ASE over the training data: ", ASE)
            ASEtrains.append(ASE)
            print("ASE over the testing data: ", ASEtest)
            ASEtests.append(ASEtest)

    plt.xlabel('d features')
    plt.ylabel('ASE')
    plt.title('Training data')
    plt.plot([2,4,6,8,10], ASEtrains, 'ro')
    plt.axis([0, 12, 10, 30])
    plt.show()

    plt.xlabel('d features')
    plt.ylabel('ASE')
    plt.title('Testing data')
    plt.plot([2,4,6,8,10], ASEtests, 'ro')
    plt.axis([0, 12, 10, 30])
    plt.show()
