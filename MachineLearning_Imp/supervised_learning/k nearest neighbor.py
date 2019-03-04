import numpy as np
from MachineLearning_Imp.utils.misc import euclidean_distance

class KNN(object):
    def __init__(self, k=5):
        self.v_k = k

    def getMaxCount(self, labels):
        """
        Get the maxmount of labels
        """
        counts = np.bincount(labels.astype('int'))
        return counts.argmax()

    def predict(self, x_train, y_train, x_test):
        y_predict = np.empty(x_test.shape[0])
        for index, value in enumerate(x_test):
            # 返回距离未知点最近的K个点的下标，升序排列
            idx = np.argsort(euclidean_distance(value, x) for x in x_train)[:self.v_k]
            # 获取这些点对应的类别
            neighbor = np.array([y_train[i] for i in idx])
            # 获得未知点所属的类别
            y_predict[index] = self.getMaxCount(neighbor)
        return y_predict