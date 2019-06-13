import sys
import numpy as np
import pandas
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 11,max_depth=25, min_samples_leaf=3)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


def loadTrain(filepath, validation=.1):
        dataframe = pandas.read_csv(filepath,sep='\t',encoding='utf-8')
        dataframe = dataframe.fillna(0.)
        data = dataframe.values
        # data = np.random.shuffle(data) #shuffle data

        vd = int(validation * len(dataframe))
        xTrain = data[vd:, 2:]
        yTrain = data[vd:, 1]
        # xTrain = data[:, 2:]
        # yTrain = data[:, 1]

        xValid = data[:vd, 2:]
        yValid = data[:vd, 1]

        xTrain = xTrain.astype('float64')
        xValid = xValid.astype('float64')

        yTrain = yTrain.astype('int')
        yValid = yValid.astype('int')

        xTrain = normalize(xTrain, axis=0, norm='max')
        xValid = normalize(xValid, axis=0, norm='max')

        # return (xTrain, yTrain), (xValid, yValid)
        return xTrain, xValid, yTrain, yValid


def read_for_testing(filename):
    dataframe = pandas.read_csv(filename,sep='\t',encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values

    testing_id = data[:, 0]
    testing_features = data[:, 1:]
    testing_features = testing_features.astype('float64')
    testing_features = normalize(testing_features, axis=0, norm='max')

    return (testing_features,testing_id)


# Driver code
def main():

    # Building Phase
    # data = importdata()
    X_train, X_test, y_train, y_test = loadTrain(sys.argv[1])
    testData = read_for_testing(sys.argv[2])

    test = testData[0]
    labels = testData[1]

    dt = train_using_gini(X_train, X_test, y_train)
    print("Results Using Gini Index:")

    preds = dt.predict(test)
    for i,pred in enumerate(preds):
        print('{},{}'.format(labels[i],pred))


# Calling main function
if __name__=="__main__":
    main()
