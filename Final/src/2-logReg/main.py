import sys
import scipy
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



def loadTrain(filepath, validation=.2):
        dataframe = pd.read_csv(filepath,sep='\t',encoding='utf-8')
        dataframe = dataframe.fillna(0.)
        xTrain, xValid, yTrain, yValid = train_test_split(dataframe.drop('class',axis=1),dataframe['class'], test_size=0.30,random_state=101)

        return (xTrain, yTrain), (xValid, yValid)


def read_for_testing(filename):
    dataframe = pd.read_csv(filename,sep='\t',encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values

    testing_id = data[:, 0]
    testing_features = data[:, 1:]
    testing_features = testing_features.astype('float64')

    return (testing_features,testing_id)


def main():
    train, test = loadTrain(sys.argv[1])

    X = train[0]
    X = X.drop('#defLine',axis=1)
    Y = train[1]

    Xtest = test[0]
    ids = Xtest['#defLine'].values

    Xtest = Xtest.drop('#defLine', axis=1)
    Ytest = test[1]


    testData = read_for_testing(sys.argv[2])
    test = testData[0]
    labels = testData[1]

    logreg = LogisticRegression()
    logreg.fit(X,Y)

    predictions = logreg.predict(test)

    for i,prediction in enumerate(predictions):

        print('{},{}'.format(labels[i],prediction))


if __name__ == '__main__':
    main()
