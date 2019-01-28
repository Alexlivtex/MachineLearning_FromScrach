"""
This file holds the basic implementation of algorithm linearRegression
"""
from __future__ import absolute_import
import numpy as np

class LinearRegression(object):
    """
    Need three parameters to construct the Regression object
    iterations_count : The total iterations that you want to run for the training phase
    learningRate : Learning rate of the training process
    lambdaRate   : The regulation parameter
    """
    def __init__(self, iterations_count, learningRate, lambdaRate):
        self.v_iterations_count = iterations_count
        self.v_learningRate = learningRate
        self.v_lambdaRate = lambdaRate

    def weigths_init(self, featuresCount):
        """
        :param featuresCount: the total feature count of the training data
        :return: None, init the weights to range[-1, 1].
        """
        self.weigths = np.random.uniform(-1, 1, (featuresCount, ))

    def training(self, x_train, y_train):
        """
        :param x_train : train data
        :param y_train : train target
        :return: None
        """
        np.insert(x_train, 0, 1, axis=1)
        self.loss = []
        self.weigths_init(x_train.shape[1])

        for i in range(self.v_iterations_count):
            predicted_y = x_train.dot(self.weigths)

            MeanSquareError = np.mean(0.5*(y_train - predicted_y)**2) + self.v_lambdaRate*0.5*self.weigths.T.dot(self.weigths)
            self.loss.append(MeanSquareError)
            grad_w = -(y_train - predicted_y).dot(x_train) + self.weigths*self.v_lambdaRate
            self.weigths = self.weigths - self.v_learningRate * grad_w
            print("Current loss is {} for iteration {}".format(MeanSquareError, i))

    def predict(self, x_test):
        X = np.insert(x_test, 0 ,1, axis=1)
        y_test = X.dot(self.weigths)
        return y_test
