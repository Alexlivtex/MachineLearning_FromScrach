import numpy as np

#Defined the sigmoid function of activation
class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def getGradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
