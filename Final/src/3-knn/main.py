import sys
import pandas
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


def loadTrain(filepath, validation=.1):
        dataframe = pandas.read_csv(filepath,sep='\t',encoding='utf-8')
        dataframe = dataframe.fillna(0.)
        data = dataframe.values
        np.random.shuffle(data) #shuffle data

        vd = int(validation * len(dataframe))
        xTrain = data[vd:, 2:]
        yTrain = data[vd:, 1]

        xValid = data[:vd, 2:]
        yValid = data[:vd, 1]

        xTrain = xTrain.astype('float64')
        xValid = xValid.astype('float64')

        yTrain = yTrain.astype('int')
        yValid = yValid.astype('int')

        xTrain = normalize(xTrain, axis=0, norm='max')
        xValid = normalize(xValid, axis=0, norm='max')

        return (xTrain, yTrain), (xValid, yValid)


def read_for_testing(filename):
    dataframe = pandas.read_csv(filename,sep='\t',encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values

    testing_id = data[:, 0]
    testing_features = data[:, 1:]
    testing_features = testing_features.astype('float64')
    testing_features = normalize(testing_features, axis=0, norm='max')


    return (testing_features,testing_id)


def main():
    if(len(sys.argv) < 2):
        print("Must include filenames and LearningRate as arguments")
    else:
        train, validation = loadTrain(sys.argv[1])
        testData = read_for_testing(sys.argv[2])

        test = testData[0]
        labels = testData[1]

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(train[0],train[1])

        predictions= model.predict(test)


        for i,prediction in enumerate(predictions):
            print('{},{}'.format(labels[i],prediction))


if __name__ == '__main__':
    main()
