"""
This file holds the implemention of KNN algorithm
"""

import math
import numpy as np

def euclidean_distance(x1, x2):
    """
    Calculate the euclidean distance
    :param x1: 1st point
    :param x2: 2nt point
    :return: The distance between two points
    """
    assert len(x1) == len(x2)
    distance = 0
    for i in range(len(x1)):
        distance += math.pow(x1[i] - x2[i], 2)
    return math.sqrt(distance)

class KNN(object):
    def __init__(self, k = 5):
        self.v_k = k

    def getMaxCount(self, labels):
        """
        Get the max part of the labels
        :param labels: The labels set in the train set
        :return: The max proportion of the labels
        """
        counts = np.bincount(labels.astype('int'))
        return counts.argmax()

    def predict(self, x_train, y_train, x_test):
        y_predict = np.empty(x_test.shape[0])
        for index, value in enumerate(x_test):
            idx = np.argsort(euclidean_distance(value, x) for x in x_train)[:self.v_k]
            neighbor = np.array([y_train[i] for i in idx])
            y_predict[index] = self.getMaxCount(neighbor)
        return y_predict
