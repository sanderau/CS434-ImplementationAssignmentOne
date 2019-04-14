import sys
import numpy as np
from numpy.linalg import inv

if(len(sys.argv) < 3):
    print("Must include filenames as arguments")

else:
    # training data
    X = np.loadtxt(sys.argv[1], usecols=range(0,13)) #features
    Y = np.loadtxt(sys.argv[1], usecols=range(13,14)) #desired outputs
    Y = np.matrix(Y)
    Y = Y.transpose()

    # dummyCol = np.ones((433,1)) #dummy variable column
    # dummyX = np.append(dummyCol, X, axis = 1) #matrix with dummy column

    XT = X.transpose()
    XTX = XT.dot(X)
    XTXinv = inv(XTX)
    W = XTXinv.dot(XT.dot(Y))

    WT = W.transpose()
    WTxXT = WT.dot(XT)

    SE = np.square(Y - WTxXT.transpose())
    ASE = SE.sum() / 443


    # testing data
    Xtest = np.loadtxt(sys.argv[2], usecols=range(0,13)) #features
    Ytest = np.loadtxt(sys.argv[2], usecols=range(13,14)) #desired outputs
    Ytest = np.matrix(Ytest)
    Ytest = Ytest.transpose()

    # dummyCol = np.ones((74,1)) #dummy variable column
    # Xtest = np.append(dummyCol, Xtest, axis = 1) #matrix with dummy column

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
