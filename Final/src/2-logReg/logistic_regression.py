import os
import sys
import numpy as np

feature_103 = True # use all features, or 103 features

#returns training data in matrix
def load_train():
    if(feature_103):
        fp = open('../data/feature103_Train.txt', 'r') #open file where data is located
    else: 
        fp = open('../data/featuresall_train.txt', 'r')

    matrix = []
    good_data = False #skip first line because we dont need it

    for line in fp:
        
        x = [] # this will hold xi
        i = 0 
        y = 0

        if(good_data):
            for word in line.split():
                if(i == 0):
                    pass
                elif(i == 1):
                  y = word #y is the last value in the matrix, add at the end
                else:
                    x.append(word)
                i+=1
            
            x.append(y)
            matrix.append(x)

        else:
            good_data = True

    numpy_matrix = np.array(matrix)
    numpy_matrix = numpy_matrix.astype(np.float)

    return numpy_matrix

#returns training data in matrix
def load_test():
    if(feature_103):
        fp = open('../data/features103_test.txt', 'r') #open file where data is located
    else: 
        fp = open('../data/featuresall_test.txt', 'r')

    matrix = []
    good_data = False #skip first line because we dont need it

    for line in fp:
        
        x = [] # this will hold xi
        i = 0 
        y = 0

        if(good_data):
            for word in line.split():
                if(i == 0):
                    pass
                elif(i == 1):
                  y = word #y is the last value in the matrix, add at the end
                else:
                    x.append(word)
                i+=1
            
            x.append(y)
            matrix.append(x)

        else:
            good_data = True
    
    numpy_matrix = np.array(matrix)
    numpy_matrix = numpy_matrix.astype(np.float)


    return numpy_matrix

def train(data, learning_rate, iterations):
    W = np.zeros(len(data[0])-1)
    
    while(iterations):
        
        d = np.zeros(len(data[0])-1)

        for i in range(len(data)):
            # get xi and yi
            x = data[i][0:len(data[0])-1]
            y = data[i][len(data[0])-1]

            #calculate
            dot = np.dot(-W.T, x)
            denom = np.longdouble(1 + np.exp(dot))
            y_hat = np.longdouble(1/denom)

            if y_hat >= .5:
                y_hat = 1

            error = y - y_hat
            d = d + (error * x)

        W = W + (learning_rate * d)
        iterations -= 1

    return W

# use calculated weight to determine test data
def test(w, test_data):
    pkf = 0 #pseudoknot free
    pkp = 0 #psuedoknot present

    for i in range(len(test_data)):
        #grab just X
        x = test_data[i]

        #guess y
        dot = np.dot(-w.T, x)
        denom = np.longdouble(1 + np.exp(dot))
        y_hat = np.longdouble(1/denom)

        if(y_hat):
            pkf += 1
        else:
            pkp += 1

    print("pkf: " + str(pkf))
    print("pkp: " + str(pkp))
    

def main():
    train_data = load_train()
    test_data = load_test()

    W = train(train_data, 1, 10)

    test(W, test_data)

main()
